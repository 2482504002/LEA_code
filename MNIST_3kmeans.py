import numpy as np
import struct
 

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。

    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
 
 
    return images #这是num_row*num_col的ndarray
 
 
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
 
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
 
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


import numpy as np

class Kmeans:
    def __init__(self,k,maxiter,data,labels):
        self.k = k  #k是簇的数目
        self.maxiter=maxiter  #最大迭代次数
        self.data=data  
        #输入的数据，是一个(m,n*n)的数组，m代表有m个测试图，n*n是特征。
        #（例如由图片28*28二维矩阵转化为这个784一维矩阵）
        self.labels=labels  #标签文件，也是一个对应位置的一维矩阵。
        self.distances=np.zeros((self.data.shape[0],self.k))
        #data.shape[0]代表测试图数目（前文注释的m）。
        #data.shape[1]代表测试图特征（前文注释的n*n）。
        #创建个空的(m,k)数组，用来保存距离。
        self.centre=np.zeros((self.k,self.data.shape[1]))
        #创建个空的(k,n*n)数组，用来保存中心点。
        
    def get_distances(self):  
        #计算每幅图到每个中心图距离的函数，算得的距离保存为一个(m,k)数组。
        #这个数组在m上的索引顺序与data相同，即还是按照原来的顺序对应图片。
        for i in range(self.data.shape[0]):  #对每幅图进行计算。
            distance_i=((np.tile(self.data[i],(self.k,1))-self.centre)**2).sum(axis=1) ** 0.5
            #计算每个点到中心点的距离，得到一个长度k的一维数组。
            self.distances[i]=distance_i  
            #将得到的一维数组放在distances的对应位置，k轴从0到k-1。
            
    def get_centre(self): #初始化中心点，并初始化分类数组。
        self.classifications=np.random.randint(0,self.k,(self.data.shape[0]))
        #创建一个(m,)的分类数组，里面填充[0,k)的随机整数，代表每个图的初始化聚类。
        for i in range(self.k):
            self.classifications[i]=i
            #防止出现空的聚类使后面中心点计算报错。
    
    
    def classify(self):  #分类的函数。
        new_classifications=np.argmin(self.distances,axis=1)
        #计算【距离数组里每一行的最小值的索引】所组成的一维数组。
        if any(self.classifications-new_classifications):
            self.classifications=new_classifications
            #如果得到的一维数组与之前的分类数组不完全相同，则该数组作为新分类数组。
            return 1
            #返回值控制外部循环。
        else:
            return 0
        #如果完全相同就跳过，不用再替换了。返回值控制外部循环。
        
    def update_centre(self):  #更新中心点的函数。
        for i in range(self.k):
            self.centre[i] = np.mean(self.data[self.classifications==i], axis=0 )
            #每个聚类的中心点是【所有标签为该聚类的点的中心】，即数据轴的平均值。
    def work(self):  #kmeans的计算函数。
        self.get_centre() #先初始化中心点。
        for i in range(self.maxiter):  #控制次数。
            self.update_centre()  #更新中心点。
            self.get_distances()  #求距离。
            if(not self.classify()):  #根据距离分类。
                break
            #如果分类不变化则停止for循环。

def kmeans(m,features):  
    # 训练集文件，右边是文件地址
    print("Start Kmeans......")
    train_images_idx3_ubyte_file = './data/MNIST/raw/train-images-idx3-ubyte'
    # 训练集标签文件
    train_labels_idx1_ubyte_file = './data/MNIST/raw/train-labels-idx1-ubyte'

    # 测试集文件
    test_images_idx3_ubyte_file = './data/MNIST/raw/t10k-images-idx3-ubyte'
    # 测试集标签文件
    test_labels_idx1_ubyte_file = './data/MNIST/raw/t10k-labels-idx1-ubyte'

    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)

    
    #m = 1500  # 创建一个读入数据的数组，部分选取图片进行训练。m为选取数目。
    n_clusters=30 #聚类的数目，即k值
    trainingMat = np.zeros((m, 784))  
    #初始化存放部分选取图片的数组并拉直，即(m,n*n)的数组。置为0。
    part_train_labels=train_labels[0:m]  #直接截取出存放这部分图片标签的数组。
    for i in range(m):
        for j in range(features):
            for k in range(28):
                trainingMat[i, 28*j+k] = train_images[i][j][k]
    #将前面m张图片赋给存放部分图片的数组。

    a=Kmeans(n_clusters,300,trainingMat,part_train_labels)
    a.work()

    label_num = np.zeros((n_clusters, 10))
    for i in range(a.classifications.shape[0]):
        pred = int(a.classifications[i])
        truth = int(part_train_labels[i])
        label_num[pred][truth] += 1
    ## 查看KNN label---> number label的对应关系
    label2num = label_num.argmax(axis=1)       
    set( label2num ) ## 看下分类是否覆盖10个数字
    train_preds = np.zeros(part_train_labels.shape)
    for i in range(train_preds.shape[0]):
        train_preds[i] = label2num[a.classifications[i]]

    print("训练数据上的精度：{}".format(np.sum(train_preds==part_train_labels) / part_train_labels.shape[0]))
    return train_images, train_preds

import pickle


if __name__=="__main__":
    m=60000
    features=28
    data,labels=kmeans(m,features)

    with open('MNIST_data.pkl', 'wb') as f1:
        pickle.dump(data, f1)
    with open('MNIST_label.pkl', 'wb') as f2:
        pickle.dump(labels, f2)
    '''with open('MNIST_label.pkl', 'rb') as f:
        loaded_variable1 = pickle.load(f)
    with open('MNIST_data.pkl', 'rb') as f2:
        loaded_variable2 = pickle.load(f2)'''

    a=1