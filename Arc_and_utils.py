import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc
# import matplotlib.pyplot as plt




def extend_df(df,filling = -999):
    for i in range(1,17):
        df['ratio_vol_app__count_app_{}'.format(i)] = df['vol_app_{}'.format(i)]/df['count_app_{}'.format(i)]
        df['ratio_vol_app__count_app_{}'.format(i)].fillna(filling)
    content_costs = df[['content_cost_m1','content_cost_m2','content_cost_m3']].fillna(filling)
    all_costs = df[['all_cost_m1', 'all_cost_m2', 'all_cost_m3']].fillna(filling)
    content_costs['all'] = content_costs['content_cost_m1'] + content_costs['content_cost_m2'] + content_costs['content_cost_m3']
    all_costs['all'] = all_costs['all_cost_m1'] + all_costs['all_cost_m2'] + all_costs['all_cost_m3']

    df['ratio_content_cost__all_cost'] = content_costs['all']/all_costs['all']
    df['ratio_content_cost__all_cost'] = df['ratio_content_cost__all_cost'].fillna(filling)
    return df

class Music_Net(torch.nn.Module):

    def __init__(self, int_size,contin_size):
        super(Music_Net, self).__init__()

        self.net_layer = torch.nn.Sequential(

            torch.nn.Linear(1024, 2048),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2048, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 2),
            torch.nn.Softmax()

        )

        self.normalizer = torch.nn.LayerNorm(contin_size)
        self.embed_cont = torch.nn.Linear(contin_size, 512)
        self.embed_int = torch.nn.Linear(int_size, 512)
        self.activate = torch.nn.ReLU()

    def forward(self, int_input, contin_input):

        contin_input = self.normalizer(contin_input)
        contin_input = self.embed_cont(contin_input)

        int_input = self.embed_int(int_input)

        input_ = torch.cat([contin_input, int_input],dim=1)
        input_ = self.activate(input_)

        return self.net_layer(input_)

class Music_dataset(Dataset):

    def __init__(self,dataframe, target_colname, cont_colnames,binary_colnames):
        '''

        loading data to Dataset

        :param dataframe: pandas.Dataframe, already prepared and preprocessed
        :param target_colname: --str, name of target column
        contin_colnames - list of str, names of continuous columns, which must be normilized in net
        '''



        self.int_features = dataframe[list(set(binary_colnames).intersection(set(dataframe.columns)))].values.tolist()

        self.contin_features = dataframe[list(set(cont_colnames).intersection(set(dataframe.columns)))].values.tolist()
        self.targets = dataframe[target_colname].values.tolist()

    def __getitem__(self, item):
        return self.int_features[item],self.contin_features[item], self.targets[item]
    def __len__(self):
        return len(self.targets)

def loading_data(dataframe,target_colname,con_colnames, binary_colnames,batch_size = 128, shuffle = True):

        return DataLoader(Music_dataset(dataframe, target_colname,con_colnames,binary_colnames), shuffle=shuffle, batch_size=batch_size,drop_last = True)


def train(epoch,data_loader, optimizer, criterion, net,values_list,epoch_list,print_every = 40,
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    for sample_id,(int_features,con_features, targets) in enumerate(data_loader):

        net.train()
        optimizer.zero_grad()

        int_input = Variable(torch.cat(int_features)).float().view(batch_size,-1).to(device)
        con_input = Variable(torch.cat(con_features)).float().view(batch_size,-1).to(device)

        target =  Variable(targets).long().to(device)

        output = net(int_input, con_input)
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()

        if sample_id % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, sample_id, len(data_loader),
                100. * sample_id / len(data_loader), loss.data.item()))
            epoch_list.append(sample_id / len(data_loader) + (epoch-1)*len(data_loader))
            values_list.append(loss.data.item())
    return loss.data.item

def test_model(net, test_data, target_name, binary_colnames, cont_colnames):

    targets = test_data[target_name].values
    test_data = test_data.drop(columns = [target_name]).values


    net.eval()
    output = net(Variable(torch.Tensor(test_data[binary_colnames].values)),
                 Variable(torch.Tensor(test_data[cont_colnames].values))).cpu().numpy().reshape(targets.shape)
    return auc(output[:,1], targets)



def save(model, path = 'trained_models/pytorch_model.pt'):

    '''

    saving pyTorch model to .pt

    :param model: torch.nn.Module, net to be saved
    :param path: --str, path where to save
    :return:None
    '''
    torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # with open('Contin_features', 'r') as f:
    #     contin_colanmes = f.readlines()
    #
    # contin_colanmes += ['ratio_content_cost__all_cost'] + ['ratio_vol_app__count_app_{}'.format(x) for x in range(1, 17)]
    # contin_colnames = [x.replace('\n', '') for x in contin_colanmes]

    binary_features = [
        'tp_flag',
        'block_flag',
        'service_1_flag',
        'service_2_flag',
        'service_3_flag',
        'is_obl_center',
        'is_my_vf',
        'service_P_flag',
        'service_7_flag',
        'service_9_flag'
    ]

    epochs = 128
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataframe = pd.read_csv('/home/yevhen/Vodafone_competion/train_music.csv')
    dataframe = extend_df(dataframe)
    dataframe = dataframe.drop(columns = ['id', 'sim_count'])
    dataframe = dataframe.fillna(0)

    dataframe, valid_dataframe  = train_test_split(dataframe,test_size=0.2, random_state=42)

    contin_colnames = list(set(dataframe.columns) - set(binary_features)) + ['ratio_content_cost__all_cost']


    cont_size = len(list(set(contin_colnames).intersection(set(dataframe.columns))))
    int_size = len(list(set(binary_features).intersection(set(dataframe.columns))))

    net = Music_Net(int_size, cont_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    data_loader = loading_data(dataframe,'target',contin_colnames, binary_features,batch_size)

    values_list = []
    epoch_list = []

    for epoch in range(epochs):
        train(epoch+1,data_loader,optimizer,criterion,net,values_list,epoch_list)

    save(net, 'norm_fillZero_Adam128.pt')

    print(test_model(net,valid_dataframe,'target',binary_features,contin_colnames))
    # plt.scatter(epoch_list, values_list)
    # plt.plot(epoch_list, values_list)
    # plt.show()
