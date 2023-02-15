import torch
from torch.nn import functional as F
import os
# from datasets import VOCDetection
#from datasets_construction import ConstructionDataset
from datasets_construction_bgSeperated import ConstructionDataset
from configurations import args
# from tensorboardX import SummaryWriter
from torchvision import transforms
from models import EventNet
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import pickle

file = 'logs/{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.patch_size, args.ratio, args.text)
file='/media/mst/Backup/PhD/Scripts/HAR/logs/paper_checkpoints/construction_resnet50_128_2.0_mod_lr_5e-4_entire_no_bg/'

checkpoint_file = file + '/checkpoints'
log_file = file + '/tensorboard'
 
def calc_metric(outputs, targets, score_thres = 0.5):

    return accuracy_score(outputs, targets), recall_score(outputs, targets, average='micro')
    
def train():
    # logger = SummaryWriter(log_file)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    
    start_epoch = 0
                    
    # resume training
    if not args.resume == "":
        
        checkpoint = torch.load(args.resume)
        #print(args.resume)
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "module.final_layer" not in k}        
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

        #model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optim'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        #start_epoch = 0 #checkpoint['epoch']

    best_train_acc, best_val_acc = 0, 0

    with open(checkpoint_file + '/training_log.txt', 'w') as f:
        f.write("")
        
    for epoch in range(start_epoch, args.max_epoch):

        # train
        model.train()
        total_train_correct, total_train_recall, total_train_loss, total_train_examples = 0, 0, 0, 0
        # loop over training set:
        for i, data in enumerate(train_data_loader):
            

            im_indices, bbs, cbs, classes, _ = [x.cuda() for x in data]
            classes = classes.long()
            output, _ = model(bbs, cbs)
            loss = F.cross_entropy(output, classes)

            #loss = F.binary_cross_entropy_with_logits(output, classes)
            total_train_loss += loss.item()

            #correct, recall = calc_metric(torch.argmax(output), classes, 0.5)
            correct, recall = calc_metric(torch.argmax(output, dim = 1).cpu().numpy(), classes.cpu().numpy())

            total_train_recall += recall
            total_train_correct += correct
            total_train_examples += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log training process
            if (i + 1) % args.log_interval == 0:
                train_rec = total_train_recall / total_train_examples
                train_acc = total_train_correct / total_train_examples
                train_loss = total_train_loss / total_train_examples
                print('train: epoch: {},  iteration: {}/{}, train_acc: {:.4f}, train_rec: {:.4f}, train_loss: {:.4f}'.format(
                        epoch, i+1, len(train_data_loader), train_acc, train_rec, train_loss))
                with open(checkpoint_file + '/training_log.txt', 'a') as f:
                    f.write('train: epoch: {},  iteration: {}/{}, train_acc: {:.4f}, train_rec: {:.4f}, train_loss: {:.4f}\n'.format(
                        epoch, i+1, len(train_data_loader), train_acc, train_rec, train_loss))
                # logger.add_scalar('train/acc', train_acc, epoch*len(train_data_loader) + i)
                # logger.add_scalar('train/loss', train_loss, epoch*len(train_data_loader) + i)
                
            if (i + 1) % 20 == 0:    
                   torch.save({'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}, checkpoint_file + '/current.pth')

        # validate
        if epoch % args.val_step == 0:
            model.eval()
            total_val_correct, total_val_recall, total_val_loss, total_val_examples = 0, 0, 0, 0
            with torch.no_grad():
                # loop over val set:
                for data in tqdm(val_data_loader):
                    im_indices, bbs, cbs, classes, _ = [x.cuda() for x in data]
                    
                    classes = classes.long()

                    output, _ = model(bbs, cbs)

                    loss = F.cross_entropy(output, classes)
                    #loss = F.binary_cross_entropy_with_logits(output, classes)
                    total_val_loss += loss.item()

                    correct, recall = calc_metric(torch.argmax(output, dim = 1).cpu().numpy(), classes.cpu().numpy())
                    total_val_recall += recall
                    total_val_correct += correct
                    total_val_examples += 1
                    
            val_rec = total_val_recall / total_val_examples
            val_acc = total_val_correct / total_val_examples
            val_loss = total_val_loss / total_val_examples
            print('\n------>val: epoch: {}, val_acc: {:.4f}, val_rec: {:.4f}, val_loss: {:.4f}\n'.format(
                            epoch, val_acc, val_rec, val_loss))
            with open(checkpoint_file + '/training_log.txt', 'a') as f:
                f.write("--------------- TEST ---------------------\n")
                f.write('val: epoch: {}, val_acc: {:.4f}, val_rec: {:.4f}, val_loss: {:.4f}\n'.format(
                            epoch, val_acc, val_rec, val_loss))
                f.write("------------------------------------------\n")

            # logger.add_scalar('val/acc', val_acc, epoch)
            # logger.add_scalar('val/loss', val_loss, epoch)

            # save weight
            if val_acc > best_val_acc:
                best_val_acc = val_acc 
                torch.save({'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch}, checkpoint_file + '/best_val_acc.pth')

        torch.save({'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}, checkpoint_file + '/current.pth')

def evaluate(exclude_bg, threshold = 0.75):
    from utils import plot_confusion_matrix, plot_multi_confusion_matrix

    checkpoint = torch.load(checkpoint_file + '/current.pth')
    #checkpoint = torch.load(checkpoint_file + '/current.pth')
    model.load_state_dict(checkpoint['model'])

    model.eval()

    total_val_correct, total_val_recall, total_val_loss, total_val_examples = 0, 0, 0, 0
    predicted_labels, gt_labels = [], []

    count = {}

    with torch.no_grad():
        # loop over val set:
        for data in tqdm(val_data_loader):
            im_indices, bbs, cbs, classes, _ = [x.cuda() for x in data]
            classes = classes.long()

            output, _ = model(bbs, cbs)

            for l in classes:
                if l.int().item() not in count:
                    count[l.int().item()] = 1
                else:
                    count[l.int().item()] += 1       
                    
            loss = F.cross_entropy(output, classes)
            #loss =  F.binary_cross_entropy_with_logits(output, classes)
            
            total_val_loss += loss.item()
            # print('outputoutputoutputoutputoutputoutputoutputoutputoutputoutputoutputoutput')
            # print(output)
            output = F.softmax(output, -1)
            prob, predict = torch.max(output, dim = 1)
            
            # prob is batch_size x n_class
            # Threshold background
            predict[prob <= threshold] = 3

            #print()
            correct, recall = calc_metric(predict.cpu().numpy(), classes.cpu().numpy())
            
            total_val_recall += recall
            total_val_correct += correct
            total_val_examples += 1

            predicted_labels.extend(predict.tolist())
            gt_labels.extend(classes.long().tolist())
            
    val_rec = total_val_recall / total_val_examples
    val_acc = total_val_correct / total_val_examples
    #print(val_acc, val_rec)
    #print(predicted_labels)
     

    with open('listfile.data', 'wb') as filehandle:
    # store the data as binary data stream
        pickle.dump(predicted_labels, filehandle)
     
     
     
     
    #with open('listfile.txt', 'w') as filehandle:
      #  filehandle.writelines("%s\n" % place for place in predicted_labels)
    #myclasses=['Nailing','Hammering','Screwing', 'background']
    
    myclasses=[ 'background','Cutting','Nailing','Watering', "Bg"]
    if exclude_bg:
        myclasses=[ 'Cutting','Nailing','Watering', "Bg"]
    # print(gt_labels)
    # print(predicted_labels)
    # print(myclasses)
    # print(args.text)
    
    # plt.figure()
    # plot_confusion_matrix( gt_labels, predicted_labels,classes=myclasses)  # doctest: +SKIP
    # plt.show()  # doctest: +SKIP
    
    
    plot_confusion_matrix(gt_labels, predicted_labels, classes=myclasses, text=args.text,
                            title='Confusion matrix, without normalization')

    #plot_confusion_matrix(gt_labels, predicted_labels, classes=val_dataset.activity_classes, text=args.text,
                           #title='Confusion matrix, without normalization')                           
    #plot_multi_confusion_matrix(gt_labels, predicted_labels, classes=val_dataset.activity_classes, text=args.text,
                            #title='Confusion matrix, without normalization', output = checkpoint_file)

def analyze():
    from utils import plot_confusion_matrix, plot_multi_confusion_matrix

    checkpoint = torch.load(checkpoint_file + '/current.pth')
 
    #checkpoint = torch.load(checkpoint_file + '/current.pth')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    total_val_correct, total_val_recall, total_val_loss, total_val_examples = 0, 0, 0, 0
    predicted_labels, gt_labels = [], []

    with open(checkpoint_file + '/errors.txt', 'w') as filename:
        filename.write("")
        
        
    with torch.no_grad():
        # loop over val set:
        for data in tqdm(val_data_loader):
            im_indices, bbs, cbs, classes, pid = [x.cuda() for x in data]
            classes = classes.long()

            output, _ = model(bbs, cbs)
            
            loss = F.cross_entropy(output, classes)
            #loss =  F.binary_cross_entropy_with_logits(output, classes)
            
            total_val_loss += loss.item()
            predict = torch.argmax(output, dim = 1)
            predict = predict.cpu().numpy()
            classes = classes.cpu().numpy()
            im_indices = im_indices.cpu().numpy()
            pid = pid.cpu().numpy()
            for i, p, c, pr in zip(im_indices, pid, classes, predict):
                if c ==0 and pr==1:
                #pr or c == p: continue
                    with open(checkpoint_file + '/errorsFP.txt', 'a') as filename:
                        filename.write("{}-{}-{}-{} ".format(i,p,c,pr))
                
            
    plot_confusion_matrix(gt_labels, predicted_labels, classes=0, text=args.text,
                            title='Confusion matrix, without normalization')
         #plot_confusion_matrix(gt_labels, predicted_labels, classes=val_dataset.activity_classes, text=args.text,
                       #     title='Confusion matrix, without normalization')                       
    #plot_multi_confusion_matrix(gt_labels, predicted_labels, classes=val_dataset.activity_classes, text=args.text,
                            #title='Confusion matrix, without normalization', output = checkpoint_file)


if __name__ == "__main__":

    if not os.path.isdir(file):
        os.mkdir(file)

    if not os.path.isdir(checkpoint_file):
        os.mkdir(checkpoint_file)
    num_classes = 4
    path="D:/Videos/Ceilling/video_1/TrainAndTest/"
    path="/media/mst/Backup/Videos/Ceilling/video_1/TrainAndTest/"

    val_dataset = ConstructionDataset(path, image_set='test', patch_size=args.patch_size,
                                        ratio=2.0, scale=args.scale, mask=False, num_classes = num_classes, exclude_bg = args.exclude_bg)                     
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4)  
    #create model, optimizer, scheduler
    n_classes = val_dataset.num_classes + 1
    if args.exclude_bg:
        n_classes = val_dataset.num_classes 
    model = EventNet(args.model, n_classes - 1, args.patch_size, args.use_feature_maps, args.use_self_attn, args.modified)
    model = torch.nn.DataParallel(model).cuda()

    if args.test:
        print('anayzing')
        evaluate(args.exclude_bg)
    else:



        path_train="D:/Videos/Ceilling/video_1/PolishingRoof/Detection/train/"
        path_train="/media/mst/Backup/Videos/Ceilling/video_1/PolishingRoof/Detection/train/"

        train_dataset = ConstructionDataset(path_train, image_set='train', patch_size=args.patch_size, ratio=2.0,
                                            scale=args.scale, mask=False, num_classes = num_classes, exclude_bg = args.exclude_bg)   
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=4)
        print('training')
        train()
