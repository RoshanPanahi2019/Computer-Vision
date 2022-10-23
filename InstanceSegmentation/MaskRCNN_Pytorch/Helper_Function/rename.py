# rename is optional for instance segmentation. 


import os
 
def main():
    path="/media/mst/Backup/Github/Computer-Vision/InstanceSegmentation/MaskRCNN_Pytorch/PennFudanPed/masks"
    folder = path
    for count, filename in enumerate(os.listdir(folder)):
        dst = f" {str(count)}.png"

        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
        os.rename(src, dst)
 
if __name__ == '__main__':
     
    main()
