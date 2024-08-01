import os
import torch
import argparse
import pickle

from torch.utils.data.dataloader import DataLoader

# # Import chexport dataset
# import vanilla_medical_classifier_chexpert.Testers as Test
# import vanilla_medical_classifier_chexpert.Trainers as Train
# from vanilla_medical_classifier_chexpert.Main import handle_dataset, create_base_transform
# import vanilla_medical_classifier_chexpert.DataHandling as DataHandling

# Import Covid19 model
from covid19_model import *

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, ClassECELoss
from Metrics.metrics2 import ECE, softmax
from Metrics.plots import temp_bins_plot_chexpert, ece_bin_plot_chexpert, logits_diff_bin_plot, reliability_plot_chexpert, temp_bins_plot2
from Metrics.plots import plot_temp_different_bins, ece_iters_plot2, plot_trajectory, conf_acc_diff_plot, ece_class_plot, conf_acc_diff_plot_class

from dataset import Chexpert, LogitsLabelsDataset, Covid19, HAM10000

# Import temperature scaling and NLL utilities
from temperature_scaling import binary_temperature_scale_test, set_temperature2, temperature_scale2, class_temperature_scale2, set_temperature3, bins_temperature_scale_test3, set_temperature4
from temperature_scaling import bins_temperature_scale_test4, bins_temperature_scale_test5, set_temperature5, MatrixScaling, set_temperature_binary

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parseArgs():
    save_plots_loc = '/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/plots'
    num_bins = 15
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("--test-bins", type=int, default=num_bins, dest="test_bins",
                        help='Number of test bins')
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    parser.add_argument("-acc", action="store_true", dest="acc_check",
                        help="whether to calculate ECE for each class only if accuracy gets better")
    parser.add_argument("-iters", type=int, default=1,
                        dest="temp_opt_iters", help="number of temprature scaling iterations")
    parser.add_argument("-init_temp", type=float, default=2.5,
                        dest="init_temp", help="initial temperature for temperature scaling")
    parser.add_argument("-const_temp", action="store_true", dest="const_temp",
                        help="whether to use constant temperature on all classes")
    parser.add_argument("--save-path-plots", type=str, default=save_plots_loc,
                        dest="save_plots_loc",
                        help='Path to save plots')
    parser.add_argument("--logits_path", type=str, default='chexpert',
                        dest="logits_path",
                        help='Path of saved logits')
    parser.add_argument("-bins", action="store_true", dest="bins_temp",
                        help="whether to calculate ECE for each bin separately")
    parser.add_argument("-dists", action="store_true", dest="dists",
                        help="Whether to optimize ECE by dists from uniform probability")
    parser.add_argument("--divide", type=str, default="equal_divide", dest="divide",
                        help="How to divide bins (reg/equal)")
    parser.add_argument("-single_weight", action="store_true", dest="single_weight",
                        help="Whether to use single WS/ CWS by bins")
    parser.add_argument("--method", type=str, default="weight", dest="method",
                        help="Method to use (weight/bins/class)")
    parser.add_argument("--load_model_path", type=str, default="./models", dest="load_model_path",
                        help="Path of saved matrix/vector scaling model")
    parser.add_argument("-train_mat_scaling", action="store_true", dest="train_mat_scaling",
                        help="Whether to find matrix/vector for matrix/vector scaling")
    parser.add_argument("--n_classes", type=int, default=6, help="number of classes")
    parser.add_argument("--model_name", type=str, default="vgg_bn", dest="model_name",
                        help="model name to train/test")
    parser.add_argument("--ds_name", type=str, default="chexpert", dest="ds_name",
                        help="dataset name")
    
    # Chexport args
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="training batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--new_split", type=int, default=0, choices=[0, 1], help="create new data split")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="which mode to use")
    parser.add_argument("--par", type=int, default=1, choices=[0, 1], help="use data parallel")
    parser.add_argument("--load_model", type=int, default=1, choices=[0, 1], help="load old model for training")
    parser.add_argument("--old_model_name", type=str, help="name of specific model in models directory")
    parser.add_argument("--single_label", action='store_true', dest="single_label", help="use only single label data")
    parser.set_defaults(single_label=True)

    return parser.parse_args()


def convert_one_hot_labels(labels):
    new_labels = torch.zeros(labels.shape[0])
    for sample in range(labels.shape[0]):
        new_labels[sample] = torch.argmax(labels[sample])
        
    return new_labels

def count_labels(labels):
    unique, counts = torch.unique(labels, return_counts=True, sorted=True)
    new_unique = []
    new_counts = []
    for u, c in zip(unique, counts):
        new_unique.append(int(u.item()))
        new_counts.append(int(c.item()))
    counts = [round(c / sum(new_counts) * 100, 2) for c in new_counts]
    
    return dict(zip(new_unique, counts))

def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            # batch_size, n_crops, c, h, w = data.size()

            # y_hat = net(data.view(-1, c, h, w))  # fuse batch_size and n_crops
            # logits = y_hat.view(batch_size, n_crops, -1).mean(1)  # avg over crops
            
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()
    
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error
    temp_opt_iters = args.temp_opt_iters
    const_temp = args.const_temp
    save_plots_loc = args.save_plots_loc
    init_temp = args.init_temp
    acc_check = args.acc_check
    dataset = args.ds_name
    model = args.model_name
    trained_loss = ''
    
    if args.ds_name == 'ham10000':
        logits_path = 'dataverse/dataverse_files'
    elif args.ds_name == 'chexpert':
        logits_path = 'vanilla_medical_classifier_chexpert'
    elif args.ds_name == 'covid19':
        logits_path = 'COVID-19'
    elif args.ds_name == 'cxr14':
        logits_path = 'CXR14'
    
    # root_dir = os.path.join(os.getcwd(), 'vanilla_medical_classifier_chexpert', 'DataSet')
    # root_dir = os.path.join(loss, 'DataSet')
    # base_transform = create_base_transform()
    
    # test_df = handle_dataset(args.new_split, 'test', args.single_label)

    # test_ds = DataHandling.CheXpert(root_dir, test_df, transform=base_transform, train_flag=False)

    # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # tester = Test.Tester(args=args)
    # logits_test, labels_test = get_logits_labels(test_loader, tester.model)
    
    # _, valid_df, w = handle_dataset(args.new_split, 'train', args.single_label)

    # valid_ds = DataHandling.CheXpert(root_dir, valid_df, transform=base_transform, train_flag=False)

    # validation_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=32)
    
    # trainer = Train.Trainer(args=args, w=w)
    # logits_val, labels_val = get_logits_labels(validation_loader, trainer.model)
    
    # chexpert_ds = Chexpert(type='train')
    # len_data = len(chexpert_ds)
    # len_val = int(0.1 * len_data)
    # len_test = int(0.1 * len_data)
    # len_train = int(len_data - len_val - len_test)
    # train_imgs = chexpert_ds.imgs[:len_train]
    # val_imgs = chexpert_ds.imgs[len_train:len_train + len_val]
    # test_imgs = chexpert_ds.imgs[len_train + len_val:]
    # train_labels = chexpert_ds.labels[:len_train]
    # val_labels = chexpert_ds.labels[len_train:len_train + len_val]
    # test_labels = chexpert_ds.labels[len_train + len_val:]
    
    with open(logits_path + '/logits/' + args.model_name + '/test_labels.pickle', 'rb') as handle:
        labels_test = pickle.load(handle)
    with open(logits_path + '/logits/' + args.model_name + '/test_logits.pickle', 'rb') as handle:
        logits_test = pickle.load(handle)
    with open(logits_path + '/logits/' + args.model_name + '/val_labels.pickle', 'rb') as handle:
        labels_val = pickle.load(handle)
    with open(logits_path + '/logits/' + args.model_name + '/val_logits.pickle', 'rb') as handle:
        logits_val = pickle.load(handle)
        
    batch_size = 32
    
    # model = DarkCovidNet()
    # model = torch.load('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/covid19_resnet50_epochs_50,lr_0.003,bs_32.pth')
    
    val_set = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19', 'val')
    test_set = Covid19('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/COVID-19', 'test')
    
    # valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # logits_val, labels_val = get_logits_labels(valloader, model)
    # logits_test, labels_test = get_logits_labels(testloader, model)
    
    # model = ResNet(num_classes=args.n_classes)
    # model = DenseNet(num_classes=args.n_classes)
    # model = VGG_BN(num_classes=args.n_classes)
    # model = torch.load('/mnt/dsi_vol1/users/frenkel2/data/calibration/focal_calibration-1/CXR14/ckpts/cxr14_vgg_bn_epochs_5,lr_0.0001,bs_32_15_classes_epoch_0_val_loss_1.3678251200931364_sched.pth')
    
    # val_set = CXR14(args, type='val')
    # test_set = CXR14(args, type='test')
    
    # valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # logits_val, labels_val = get_logits_labels(valloader, model)
    # logits_test, labels_test = get_logits_labels(testloader, model)
    
    if args.method == 'vs' or args.method == 'ms':
        val_set = LogitsLabelsDataset(args, logits_path, type='val')
        val_loader = DataLoader(val_set, batch_size=128, shuffle=True)
        
    # labels_test = convert_one_hot_labels(labels_test)
    # labels_val = convert_one_hot_labels(labels_val)
    
    # count_labels_val_dict = count_labels(labels_val)
    # count_labels_test_dict = count_labels(labels_test)
    
    # print('val labels: ', count_labels_val_dict)
    # print('test labels: ', count_labels_test_dict)

    ece_criterion = ECELoss(n_bins=10).cuda()
    csece_criterion = ClassECELoss(n_bins=10).cuda()
    # vector_scaling = TestVectorScaling()
    
    # before_indices, after_indices = check_movements(logits_val, const=2)
    # plot_temp_different_bins(save_plots_loc)
    # plot_trajectory(save_plots_loc)
    
    p_ece = ece_criterion(logits_test, labels_test).item()
    # p_ece2 = ece_criterion(logits_test2, labels_test2).item()
    conf_matrix, p_acc, _, predictions, confidences = test_classification_net_logits(logits_test, labels_test)
    print(conf_matrix)
    # if not os.path.isdir(save_plots_loc + '/' + dataset + '_' + model):
    #     os.mkdir(save_plots_loc + '/' + dataset + '_' + model)
    # reliability_plot_chexpert(confidences, predictions, labels_test, save_plots_loc, dataset, model, num_bins=num_bins, scaling_related='before', save=True)
    
    if dataset == 'cxr14':
        classes_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'PT', 'Hernia', 'No findings']
    elif dataset == 'ham10000':
        classes_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:
        classes_names = []
    csece, c_acc, conf_acc_diffs_before = csece_criterion(logits_test, labels_test)
    print(c_acc)
    # ece_class_plot(csece, save_plots_loc, dataset, model, trained_loss, scaling_related='before', unc=True)
    # conf_acc_diff_plot_class(conf_acc_diffs, classes_names, save_plots_loc, dataset, model, trained_loss, divide=args.divide, ds=dataset, version='before')
    
    # weights = vector_scaling(logits_val, p_acc)
    
    # Printing the required evaluation metrics
    if args.log:
        print('Pre-scaling test ECE: ' + str(p_ece))
        print('Pre-scaling test accuracy: ' + str(p_acc))

    if args.method == 'vs' or args.method == 'ms':  # Vector/Matrix scaling
        if args.method == 'vs':
            scaling = MatrixScaling(val_loader, input_size=logits_val.shape[-1], device=device, load_model_path=args.load_model_path, is_matrix=False)
        else:
            scaling = MatrixScaling(val_loader, input_size=logits_val.shape[-1], device=device, load_model_path=args.load_model_path)
        
        if args.train_mat_scaling:
            scaling.find_best_transform(args.load_model_path, save_model=True)
            
        cal_logits = scaling.evaluate(logits_test)
        ece = ece_criterion(cal_logits, labels_test).item()
        print(ece)
        
        single_temp = set_temperature2(logits_val, labels_val.long(), const_temp=True)
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test.long()).item()
        
    if args.method == 'weight':
        if args.single_weight:
            weight, single_temp = set_temperature5(logits_val, labels_val, num_bins, log=args.log)
        else:
            bins_T, single_temp, bin_boundaries, best_iter, conf_acc_diff = set_temperature4(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins, top_temp=1.2)
            
        
    elif args.method == 'bins':
        if const_temp:
            temperature = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                        init_temp=init_temp, const_temp=const_temp, log=args.log, num_bins=num_bins)
        else:                              
            bins_T, single_temp, bin_boundaries, many_samples, best_iter = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                            const_temp=const_temp, log=args.log, num_bins=num_bins)
            #bins_T2, single_temp2, bin_boundaries2, many_samples2, best_iter2 = set_temperature3(logits_val2, labels_val2, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
            #                                                                                    acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins, top_temp=1.2)
        
    elif args.method == 'binary':
        csece_temperature, single_temp = set_temperature_binary(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                                            init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins)
        best_iter = 0
        
    else:    
        if const_temp:
            temperature = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                        init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins)
        else:                              
            csece_temperature, single_temp = set_temperature2(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error,
                                                            init_temp=init_temp, acc_check=acc_check, const_temp=const_temp, log=args.log, num_bins=num_bins)
    
    """
    softmaxs = softmax(class_temperature_scale2(logits_test, csece_temperature))
    preds = np.argmax(softmaxs, axis=1)
    confs = np.max(softmaxs, axis=1)
    ece = ECE(confs, preds, labels_test, bin_size = 1/num_bins)
    """
    
    if args.method == 'weight':
        if args.single_weight:
            new_softmaxes = bins_temperature_scale_test5(logits_test, weight)
            
        else:
            new_softmaxes, ece_bin, single_ece_bin, origin_ece_bin, ece_list = bins_temperature_scale_test4(logits_test, labels_test, bins_T,
                                                                                                        args.temp_opt_iters,
                                                                                                        bin_boundaries,
                                                                                                        single_temp, best_iter, num_bins)
            conf_acc_diff_plot(conf_acc_diff, save_plots_loc, dataset, model, trained_loss, divide=args.divide, ds='val_dists_bins', version=2)                                                                                            
            bins_T2, single_temp2, bin_boundaries2, many_samples, best_iter2 = set_temperature3(logits_val, labels_val, temp_opt_iters, cross_validate=cross_validation_error, init_temp=init_temp,
                                                                                            const_temp=const_temp, log=args.log, num_bins=num_bins)
            # temp_bins_plot(single_temp, bins_T, bin_boundaries, save_plots_loc, dataset, model, trained_loss,
            #                divide=args.divide, ds='val_dists', version=2, cross_validate=cross_validation_error, y_name='Weight')
            
            temp_bins_plot2(single_temp, single_temp2, bins_T, bins_T2, bin_boundaries, bin_boundaries2, save_plots_loc, dataset,
                            model, trained_loss, divide=args.divide, ds='val_dists_bins', version=2, y_name='(1/Temperature) / Weight')
        _, _, _, predictions, confidences = test_classification_net_logits(new_softmaxes, labels_test, is_logits=False)
        reliability_plot_chexpert(confidences, predictions, labels_test, save_plots_loc, dataset, model, num_bins=num_bins, scaling_related='after_dists', save=True, single=args.single_weight)
        _, _, _, predictions, confidences = test_classification_net_logits(temperature_scale2(logits_test, single_temp), labels_test)
        reliability_plot_chexpert(confidences, predictions, labels_test, save_plots_loc, dataset, model, num_bins=num_bins, scaling_related='after_single', save=True)
        ece = ece_criterion(new_softmaxes, labels_test, is_logits=False).item()
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test).item()
        
        csece, _, conf_acc_diffs_after = csece_criterion(new_softmaxes, labels_test, is_logits=False)
        ece_class_plot(csece, save_plots_loc, dataset, model, trained_loss, scaling_related='after_dists')
        conf_acc_diff_plot_class(conf_acc_diffs_before, conf_acc_diffs_after, classes_names, save_plots_loc, dataset, model, trained_loss, divide=args.divide, ds=dataset, version='before_and_0.00-0.34after_dists')
    
    elif args.method == 'bins':
        temp_bins_plot_chexpert(single_temp, bins_T, bin_boundaries, save_plots_loc, dataset, model,
                       divide=args.divide, ds='val', version=2, cross_validate=cross_validation_error)
        #temp_bins_plot2(single_temp, single_temp2, bins_T, bins_T2, bin_boundaries, bin_boundaries2, save_plots_loc, dataset, model, 
        #                trained_loss, divide=args.divide, ds='val_two_models', version=2)
        scaled_logits, ece_bin, single_ece_bin, origin_ece_bin, ece_list = bins_temperature_scale_test3(logits_test, labels_test, bins_T,
                                                                                                        args.temp_opt_iters,
                                                                                                        bin_boundaries, many_samples,
                                                                                                        single_temp, best_iter, num_bins)
        #scaled_logits2, ece_bin2, single_ece_bin2, origin_ece_bin2, ece_list2 = bins_temperature_scale_test3(logits_test2, labels_test2, bins_T2,
        #                                                                                                    args.temp_opt_iters,
        #                                                                                                    bin_boundaries2, many_samples2,
        #                                                                                                    single_temp2, best_iter2, num_bins)
        #ece_list = [p_ece] + ece_list
        #ece_list2 = [p_ece2] + ece_list2
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test).item()
        #ece_single2 = ece_criterion(temperature_scale2(logits_test, single_temp2), labels_test).item()
        #ece_iters_plot2(ece_single, ece_single2, ece_list, ece_list2, save_plots_loc, dataset, model, 
        #                trained_loss, divide=args.divide, ds='val_two_models_iters', version=2)
        _, _, _, predictions, confidences = test_classification_net_logits(scaled_logits, labels_test)
        reliability_plot_chexpert(confidences, predictions, labels_test, save_plots_loc, dataset, model, num_bins=num_bins, scaling_related='after_bins', save=True)
        ece_bin_plot_chexpert(ece_bin, single_ece_bin, origin_ece_bin, save_plots_loc, dataset, model,
                       divide=args.divide, ds='test', version=2)
        ece = ece_criterion(scaled_logits, labels_test).item()
        
        csece, _, conf_acc_diffs = csece_criterion(scaled_logits, labels_test)
        ece_class_plot(csece, save_plots_loc, dataset, model, trained_loss, scaling_related='after_bins')
        conf_acc_diff_plot_class(conf_acc_diffs, classes_names, save_plots_loc, dataset, model, trained_loss, divide=args.divide, ds=dataset, version='after_bins')
        
    elif args.method == 'binary':
        new_confidences, new_accuracies = binary_temperature_scale_test(logits_test, labels_test, csece_temperature, best_iter)
        ece = ece_criterion(logits_test, labels_test, accuracies=new_accuracies, confidences=new_confidences).item()
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test).item()
    
    else:
        ece = ece_criterion(class_temperature_scale2(logits_test, csece_temperature), labels_test).item()
        ece_single = ece_criterion(temperature_scale2(logits_test, single_temp), labels_test).item()
    # _, _, _, predictions, confidences = test_classification_net_logits(temperature_scale2(logits_test, single_temp), labels_test)
    # reliability_plot(confidences, predictions, labels_test, save_plots_loc, dataset, model, trained_loss, num_bins=num_bins, scaling_related='after_single', save=True)
    #_, acc, _, _, _ = test_classification_net_logits(class_temperature_scale2(logits_test, csece_temperature), labels_test)
    
    if args.log:
        print ('Post-scaling ECE (' + args.method + ' scaling): ' + str(ece))
        print ('Post-scaling ECE (Single temp scaling): ' + str(ece_single))
        print ('Post-scaling accuracy: ' + str(p_acc))
        print ('Pre-scaling ECE: ' + str(p_ece))
        
