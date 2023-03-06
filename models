 # Define the architecture of the neural network
 class Net(nn.Module):
     def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='valid')
         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='valid')
         self.dropout1 = nn.Dropout(0.25)
         self.dropout2 = nn.Dropout(0.5)
         self.fc1 = nn.Linear(9216, 128)
         self.fc2 = nn.Linear(128, 1)

     def forward(self, x):
         x = self.conv1(x)
         x = F.leaky_relu(x)
         x = self.conv2(x)
         x = F.leaky_relu(x)
         x = F.max_pool2d(x, 2)
         x = self.dropout1(x)
         x = torch.flatten(x, 1)
         x = self.fc1(x)
         x = F.leaky_relu(x)
         x = self.dropout2(x)
         output =self.fc2(x)
         #print(output.shape)
         #output = F.relu(output)
        # output = F.softmax(x, dim=1)
         #output = torch.clamp(x, 0.1, 0.8)
        # output = torch.ge(x, 0.5).float()
        # output = nn.functional.one_hot(torch.argmax(x, dim=1), num_classes=10)
         return output
         
 def mtrxbtch(batch_size):
  mtrxbtch=torch.eye(5).reshape(1,5,5)
  for i in range(batch_size-1):
    mtrxbtch=torch.cat((mtrxbtch,torch.eye(5).reshape(1,5,5)),0)
  return mtrxbtch
  
 def shuffleIm(tensor):
  mtrx = mtrxbtch(tensor.shape[0]).cuda()
  for i in range(tensor.shape[0]):
    indexes = torch.randperm(tensor.shape[1])
    tensor[i] = tensor[i][indexes]
    mtrx[i]   = mtrx[i][indexes]
  return tensor,mtrx
  
 class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(64 * 2, 64 * 4, 4, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 4, 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)
        
 class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.attention = AttentionLayer()
        self.fc1 = nn.Linear(10, 10*2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(10 * 2, 10)
        self.norm1 = nn.LayerNorm(10)
        self.norm2 = nn.LayerNorm(10)

    def forward(self, x):
        x_ = self.attention(x, x, x)
        x = x + x_
        x = self.norm1(x)
        x_ = self.fc1(x)
        x = self.activation(x)
        x_ = self.fc2(x_)
        x = x + x_
        x = self.norm2(x)

        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #self.latent_dim = 129
        
        self.fc1 = nn.Linear(129, 7*7*128)
        self.relu = nn.ReLU()
        self.conv_transpose1 = nn.ConvTranspose2d(6*128, 3*64, kernel_size=2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(3*64, 3, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = x.view(-1,6* 128, 7, 7)
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.sigmoid(x)
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #self.latent_dim = 129 
        self.fc1 = nn.Linear(129, 2*129)
        self.fc2 = nn.Linear(2*129, 5)
        self.fc3 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
      #  self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = x.transpose(1,2)
        x = self.fc3(x)
        x = x.transpose(1,2)
       # x = F.softmax(x, dim=1)
        return x
        
class ImgEncoder(nn.Module):
  def __init__(self):        
    super(ImgEncoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 3*32, kernel_size=3, stride=1, padding='valid')
    self.conv2 = nn.Conv2d(3*32, 3*64, kernel_size=3, stride=1, padding='valid')
    self.dropout1 = nn.Dropout(0.25)
    self.fc1 = nn.Linear(9216, 128)
  # self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = x.reshape(64,3,64,144)
    x = torch.flatten(x, 2)
    x = self.fc1(x)
    x = F.relu(x)
    return x        
        return x        
