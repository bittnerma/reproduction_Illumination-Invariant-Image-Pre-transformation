import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

    
def load_MNIST(batch_size = 4):
    transform = transforms.Compose(
        [transforms.ToTensor()])#,
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    classes = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')

    return trainloader,testloader,classes


from PIL import Image

import transforms as ext_transforms
from args import get_arguments

args = get_arguments()

device = torch.device(args.device)

dataset = "camvid"

args.batch_size = 2

from data.utils import median_freq_balancing

def get_data_loaders(dataset,train_batch_size,test_batch_size,val_batch_size,single_sample=False):
    print("\nLoading dataset...\n")

    print("Selected dataset:", dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)

    if single_sample:
        print("Reducing Training set to single batch of size {}".format(train_batch_size))
        train_set_reduced = torch.utils.data.Subset(train_set, list(range(0,train_batch_size)))  
    
        train_loader = data.DataLoader(
            train_set_reduced,
            batch_size=train_batch_size,
            shuffle=False,#Changed this 
            num_workers=args.workers)

    else:
        train_loader = data.DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=False,#Changed this 
            num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Test dataset size:", len(test_set))
    print("Validation dataset size:", len(val_set))
  
    class_weights = 0

    # Get class weights from the selected weighing technique    
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0  
    
    class_weights = median_freq_balancing(train_loader, num_classes)    

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding,(train_set,val_set,test_set)
