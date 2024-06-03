import torch
from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones,nn
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim

class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        #需要将weight初始化为一个Parameter对象
        self.w = Parameter(ones((1,dimensions)))#这里的1代表了一个维度，dimensions代表了特征的维度

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w #返回当前的权重

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        #计算得分，需要使用tensordot函数，这里的x是一个1*dimensions的tensor，dot之后得到一个1*2的二维tensor
        return torch.mm(x, self.w.t()).squeeze()#squeeze()函数用于去除维度为1的维度，这里的返回值是一个1*1的tensor


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #如果得分大于0，返回1，否则返回-1，但是这里的run函数返回的是一个tensor，所以需要使用item()函数将其转换为一个数值
        return 1 if self.run(x).item() >= 0 else -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """

        with no_grad():#使用no_grad()函数来关闭梯度计算，是为了加快训练速度
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            #关闭梯度计算，则使用基于不使用梯度的方法来训练模型
            #这里的dataloader是一个迭代器，每次迭代返回一个字典，字典中包含了x和label两个键值对
            #x是一个1*dimensions的tensor，label是一个数值
            #这里的label是一个数值，所以可以直接使用
            #设置一个标志位，当所有的数据都被正确分类时，flag为True，退出循环
            while True:
                flag = True
                for data in dataloader:
                    x = data['x'].view(1,-1)#view的作用是将一个tensor转换为一个1*dimensions的tensor
                    label = data['label']#这里的label是一个数值，所以可以直接使用
                    if self.get_prediction(x) != label:#如果预测错误，则更新权重
                        flag = False
                        #更新权重，self.w +=direction*magnitude
                        #direction = label*x
                        #magnitude = 1,这个代表了学习率
                        self.w += label * x #这里的x是一个1*dimensions的tensor
                if flag:
                    break




class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super(RegressionModel, self).__init__()
        self.hidden_size = 80#隐藏层的大小
        # 定义权重和偏置
        self.fc1 = Linear(1, self.hidden_size)  #使用一个线性层将输入映射到隐藏层
        self.fc2 = Linear(self.hidden_size, 1)  # 这是将隐藏层映射到输出层

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        h = relu(self.fc1(x))#使用relu激活函数
        y_pred = self.fc2(h)#这里的y_pred是一个batch_size*1的tensor，不需要再使用激活函数
        return y_pred

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        #使用损失函数来计算损失
        return mse_loss(self.forward(x),y)#这里的forward(x)是一个batch_size*1的tensor，y也是一个batch_size*1的tensor
 
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        #使用Adam优化器来优化模型
        while True:
            for data in dataloader:
                x = data['x']
                y = data['label']
                optimizer.zero_grad()#梯度清零
                loss = self.get_loss(x, y)#计算损失
                loss.backward()#反向传播
                optimizer.step()#更新权重
                #每个epoch结束后打印一下损失，如果损失小于0.001，则退出循环
            #获得所有数据的损失，当损失小于0.001时，退出循环
            if self.get_loss(dataset[:]['x'], dataset[:]['label']) < 0.001:#这里的dataset[:]['x'],[:]表示所有的数据
                break


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        #定义一个线性层，将输入映射到输出
        self.fc1 = Linear(input_size, 100)
        self.fc2 = Linear(100, 10)#这里的10代表了10个类别



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        #这里的x是一个batch_size*784的tensor
        x = relu(self.fc1(x))#使用relu激活函数
        x = self.fc2(x)
        return x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        return cross_entropy(self.run(x), y)#这里的run(x)是一个batch_size*10的tensor，y是一个batch_size*10的tensor

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)#这个是学习率调整策略
        while True:
            total_loss = 0
            for data in dataloader:
                x = data['x']
                y = data['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            if total_loss/len(dataloader)< 0.02:
                break



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        # Initialize your model parameters here
        "定义各个层的参数"
        self.hidden_size = 256
        self.num_languages = 5
        #设置三个线性层，一个输入层，两个隐藏层，一个输出层，分别将输入映射到隐藏层，隐藏层映射到隐藏层，隐藏层映射到输出层，两个隐藏层的激活函数都是relu
        #两个隐藏层的作用是为了提取特征，与只有一个隐藏层相比，两个隐藏层的模型更加复杂，可以提取更多的特征
        self.input_fc = nn.Linear(self.num_chars, self.hidden_size)
        self.hidden_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_fc = nn.Linear(self.hidden_size, self.num_languages)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size)
        self.dropout = nn.Dropout(0.5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].size(0)#提取了序列的长度
        h = torch.zeros(batch_size, self.hidden_size)#初始化隐藏层，全部为0

        # Process each character in the sequence
        for x in xs:#遍历序列中的每一个字符
            x = self.relu(self.input_fc(x))
            if batch_size > 1 and self.training:#如果batch_size大于1并且是训练模式
                x = self.batch_norm1(x)
            h = self.relu(self.hidden_fc1(h + x))#将输入和隐藏层相加，然后使用relu激活函数
            if batch_size > 1 and self.training:
                h = self.batch_norm2(h)
            h = self.relu(self.hidden_fc2(h))
            if batch_size > 1 and self.training:
                h = self.batch_norm3(h)
            h = self.dropout(h)#使用dropout函数，添加dropout层，防止过拟合

        out_logits = self.output_fc(h)
        return out_logits
    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return cross_entropy(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        epoch =0;
        while True:
            total_loss = 0
            for data in dataloader:
                xs = data['x'].permute(1, 0, 2)
                y = data['label']

                optimizer.zero_grad()
                loss = self.get_loss(xs, y)
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)#梯度裁剪

                optimizer.step()

                total_loss += loss.item()
            scheduler.step(total_loss / len(dataloader))
            print("Epoch: ", epoch, "Accuracy: ", dataset.get_validation_accuracy())
            epoch += 1
            if dataset.get_validation_accuracy() > 0.85:
                break


        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape

    input_height, input_width = input_tensor_dimensions
    weight_height, weight_width = weight_dimensions#获取输入和权重的维度

    output_height = input_height - weight_height + 1#计算输出的高度和宽度
    output_width = input_width - weight_width + 1#计算输出的高度和宽度

    Output_Tensor = torch.zeros((output_height, output_width))#初始化输出的tensor

    # Perform the convolution operation
    for y in range(output_height):
        for x in range(output_width):
            sub_tensor = input[y:y + weight_height, x:x + weight_width]#获取子tensor
            Output_Tensor[y, x] = torch.sum(sub_tensor * weight)#计算卷积
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.fc1 = Linear(26 * 26, 100)#定义一个线性层，将输入映射到隐藏层
        self.fc2 = Linear(100, output_size)#定义一个线性层，将隐藏层映射到输出层

    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)#将x展平
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        return cross_entropy(self.run(x), y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        epoch = 0
        while True:
            total_loss = 0
            for data in dataloader:
                x = data['x']
                y = data['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step(total_loss / len(dataloader))
            print("Epoch: ", epoch, "Accuracy: ", dataset.get_validation_accuracy())
            epoch += 1
            if dataset.get_validation_accuracy() > 0.98:
                break
 