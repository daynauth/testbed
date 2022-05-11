from numpy import size
import pandas as pd
import matplotlib.pyplot as plt

def plot_line():
    data = pd.read_csv('data1.csv')

    df = pd.DataFrame(data, columns = ["Epochs", "avtn1", "avtn1+", "avtn2+", "ssd_vgg16", 
        "ssd_vgg16_coco", "ssd_resnet50_coco_frozen", "ssd_resnet50_coco_unfrozen", "ssd_resnet50"])
    line = df.plot.line(x = "Epochs", y = ["avtn1", "avtn1+", "avtn2+", "ssd_vgg16", "ssd_vgg16_coco", 
        "ssd_resnet50_coco_frozen", "ssd_resnet50_coco_unfrozen", "ssd_resnet50"], ylim = (0, 0.25))

    line.set_ylabel("mAP")

    plt.savefig("graph1.png")


def plot_line2():
    data = pd.read_csv('data1.csv')

    df = pd.DataFrame(data, columns = ["Epochs", "avtn1", "avtn2", "avtn1+", "avtn2+", "ssd_vgg16", 
        "ssd_vgg16_coco", "ssd_resnet50_coco_frozen", "ssd_resnet50_coco_unfrozen", "ssd_resnet50"])


    fig, ax = plt.subplots()
    ax.plot(df['Epochs'],df['avtn1'], marker='^', label = 'avtn1')
    ax.plot(df['Epochs'],df['avtn2'], marker='^', label = 'avtn2')
    ax.plot(df['Epochs'],df['avtn1+'], marker='^', label = 'avtn1+')
    ax.plot(df['Epochs'],df['avtn2+'], marker='^', label = 'avtn2+')

    ax.plot(df['Epochs'],df['ssd_vgg16'], marker='o', label='ssd_vgg16')
    ax.plot(df['Epochs'],df['ssd_vgg16_coco'], marker='o', label = 'ssd_vgg16_coco')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')

    ax.legend(loc='lower right')
    ax.set_ylim(0, 0.25)


    plt.savefig("graph5.png")


def plot_line3():
    data = pd.read_csv('data1.csv')

    df = pd.DataFrame(data, columns = ["Epochs", "avtn1", "avtn2", "avtn1+", "avtn2+", "ssd_vgg16", 
        "ssd_vgg16_coco", "ssd_resnet50_coco_frozen", "ssd_resnet50_coco_unfrozen", "ssd_resnet50"])


    fig, ax = plt.subplots()
    ax.plot(df['Epochs'],df['avtn1'], marker='^', label = 'avtn1')
    ax.plot(df['Epochs'],df['avtn2'], marker='^', label = 'avtn2')
    ax.plot(df['Epochs'],df['avtn1+'], marker='^', label = 'avtn1+')
    ax.plot(df['Epochs'],df['avtn2+'], marker='^', label = 'avtn2+')

    ax.plot(df['Epochs'],df['ssd_resnet50_coco_unfrozen'], marker='o', label='ssd_resnet50_coco_unfrozen')
    ax.plot(df['Epochs'],df['ssd_resnet50'], marker='o', label = 'ssd_resnet50')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')

    ax.legend(loc='lower right')
    ax.set_ylim(0, 0.25)


    plt.savefig("graph6.png")

def plot_scatter1():
    y_column = 'trainable_params'
    data = pd.read_csv('data2.csv')
    df = pd.DataFrame(data, columns = ['Models', 'map', 'trainable_params', 'total_params'])
    
    fig, ax = plt.subplots()
    ax.scatter(x=df['map'],y=df[y_column],c='DarkBlue')

    ax.set_xlabel('Mean Average Precision')
    ax.set_ylabel(y_column)
    ax.set_xlim([0.1, 0.25])


    # for idx, row in df.iterrows():
    #     ax.annotate(row['Models'], (row['map'], row[y_column]))

    row = df.iloc[0]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points')

    row = df.iloc[1]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points')

    row = df.iloc[2]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points')

    row = df.iloc[3]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points')

    row = df.iloc[4]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-60,0), textcoords = 'offset points')

    row = df.iloc[5] #ssd_vgg16_coco
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-30,-100), 
        textcoords = 'offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    row = df.iloc[6] #ssd_resnet50_coco_frozen
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-50,-15), textcoords = 'offset points')

    row = df.iloc[7]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-70,5), textcoords = 'offset points')


    row = df.iloc[8] 
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-30,-100), 
        textcoords = 'offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.show()
    plt.savefig('graph2.png')

def plot_scatter2():
    y_column = 'total_params'
    data = pd.read_csv('data2.csv')
    df = pd.DataFrame(data, columns = ['Models', 'map', 'trainable_params', 'total_params'])
    
    fig, ax = plt.subplots()
    ax.scatter(x=df['map'],y=df[y_column],c='DarkBlue')

    ax.set_xlabel('Mean Average Precision')
    ax.set_ylabel(y_column)
    ax.set_xlim([0.1, 0.25])

    row = df.iloc[0]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-35,0), textcoords = 'offset points') #fine

    row = df.iloc[1]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points') #fine

    row = df.iloc[2] #avtn1+
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-20,5), textcoords = 'offset points')#fine

    row = df.iloc[3]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,1), textcoords = 'offset points') #fine

    row = df.iloc[4]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-60,0), textcoords = 'offset points') #fine

    row = df.iloc[5]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (5,-10), textcoords = 'offset points') #fine

    row = df.iloc[6]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-70,5), textcoords = 'offset points') #fine

    row = df.iloc[7]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-80,60), 
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        textcoords = 'offset points') #fine

    row = df.iloc[8]
    ax.annotate(row['Models'], (row['map'], row[y_column]), xytext = (-80,40), 
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        textcoords = 'offset points') #fine


    plt.show()
    plt.savefig('graph3.png')

def plot_bar():
    data = pd.read_csv('data2.csv')
    df = pd.DataFrame(data, columns = ['Models', 'map', 'trainable_params'])  
    df = df.set_index('Models')

    ax = df.plot( kind= 'bar' , secondary_y= 'trainable_params' , rot= 15, figsize = (20, 15), fontsize = 15)
  
    # fig, ax = plt.subplots()
    # ax.bar(x=df['Models'], height = df['trainable_params'])
    # plt.xticks(rotation=45)

    # ax.set_xlabel('Models')

    # plt.show()
    plt.savefig('graph4.png')

#plot_line()
#plot_scatter1()
#plot_bar()
plot_line3()

