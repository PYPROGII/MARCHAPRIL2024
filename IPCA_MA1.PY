from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
# --
def process_MINST_for_ML(minstData):
    counter=0
    XlistTrain=[]
    ylistTrain=[]
    XlistTest=[]
    ylistTest=[]
    for element in minstData:
        counter+=1
        if counter <60001:
            d_list=list(element)
            XlistTrain.append(d_list[:784])
            ylistTrain.append(d_list[784:])
        else:
            d_list=list(element)            
            XlistTest.append(d_list[:784])
            ylistTest.append(d_list[784:])
    #print("samples of y",ylistTrain[4:5],ylistTrain[5000:5001])
    print("ylistTest\n\n",ylistTest[1000:1008])
    #print("ylistTest",ylistTest)
    cnt_0 = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    cnt_4 = 0
    cnt_5 = 0
    cnt_6 = 0
    cnt_7 = 0
    cnt_8 = 0
    cnt_9 = 0
    for i in range(len(ylistTest)):
        if ylistTest[i][0] == b'0':
            cnt_0+=1
        if (ylistTest[i][0]) == b'1':
            cnt_1+=1
        if ylistTest[i][0] == b'2':
            cnt_2+=1
        if ylistTest[i][0] == b'3':
            cnt_3+=1
        if ylistTest[i][0] == b'4':
            cnt_4+=1
        if ylistTest[i][0] == b'5':
            cnt_5+=1
        if ylistTest[i][0] == b'6':
            cnt_6+=1
        if ylistTest[i][0] == b'7':
            cnt_7+=1
        if ylistTest[i][0] == b'8':
            cnt_8+=1
        if ylistTest[i][0] == b'9':
            cnt_9+=1
    print("counts of 0", cnt_0)
    print("counts of 1", cnt_1)
    print("counts of 2", cnt_2)
    print("counts of 3", cnt_3)
    print("counts of 4", cnt_4)
    print("counts of 5", cnt_5)
    print("counts of 6", cnt_6)
    print("counts of 7", cnt_7)
    print("counts of 8", cnt_8)
    print("counts of 9", cnt_9)
    return XlistTrain,ylistTrain,XlistTest,ylistTest
# --
def horizontal_plot(X,y,indexlist,maxwidth):
    # -- maxwidth is the number of subplots to be created
    
    fgr, ax = plt.subplots(1,maxwidth, figsize=(4,4))
    for w in range(maxwidth):
        digit=np.asarray(X[indexlist[w]])
        digit=digit.reshape(28,28)
        ax[w].imshow(digit,cmap='binary')
        ax[w].set_axis_off()                # Suppress the axis of each subplot
    fgr.tight_layout()
    #plt.suptitle("Fig3_Sample Digits")
    plt.show()
#---
def plot_digits(X,y): 
    print(type(y))
    limit = 4
    fgr, ax = plt.subplots(4,1, figsize=(4,4))# ,sharex=True, sharey=True)
    print('len(ax)',len(ax))
    for d in range(limit):
        print(int(y[d*1000][0]))       # select digits 1000 apart - print value
        digit= np.asarray(X[d*1000])   # select digit images 1000 apart
        digit=digit.reshape(28,28)
        ax[d].imshow(digit,cmap='binary')
        ax[d].set_axis_off()
    fgr.tight_layout()
    plt.suptitle("Fig3_Sample Digits")
    plt.show()
    # --------
def predict_Class( XTrain,yTrain,XTest,yTest ):
    sgdClass = SGDClassifier(random_state=42)
    sgdClass.fit(XTrain,yTrain)
    pred = sgdClass.predict(XTest)
    print("prediction", len(pred))
    print("prediction type",type(pred))
    print(pred[20])
def data_examination(data, meta):
    # - Examination of data
    print('data, shape',data.shape)
    print('meta type',type(meta))
    print("-------------Meta\----------\n")
    print(meta)
    print("---------End-Meta\----------\n")
    print("Selecting the second of the 70000 digits")
    print("data[1]")
    dg = data[1]              # -- First element of the data array

    print('\n\n')
    print('digit shape\t',dg.shape)
    print('digit length\t',len(dg)) 
    print('digit type\t',type(dg))
    print(dg)                 # -- Digit raw data
    print("sample of last 10000",data[60002],"\n\n",data[67670])
        
# ------
def main():

    data, meta = arff.loadarff('mnist_784.arff')
    data_examination(data, meta) # -- examination of a sample of the data.
    XTrain,yTrain,XTest,yTest = process_MINST_for_ML(data)
    print("Size of XTrain",len(XTrain),"yTrain",len(yTrain),"XTest",len(XTest),"yTest",len(yTest))
    predict_Class( XTrain,yTrain,XTest,yTest )
    

    #plot_digits(XTrain,yTrain)  # -- Verical Plot

    # -- Select data of specific digits to plot for a visual
    
    # -- We are getting sample indexea of 0s,6s, 8s,2s in the
    # -- Following lists. Ensure you are getting more than you will actually
    # -- display because the data is not ordered by the sequence 0 1 2 3 4 5 6 7 8 9
    index_of_0s=[]
    index_of_4s=[]
    index_of_2s=[]
    #for i in range(len(yTrain)):
    for i in range(len(yTest)):
        if (i > 100 and i < 109) or (i > 1000 and i < 1019) or (i > 2100 and i < 2111) or \
           (i > 3200 and i < 3215) or (i > 4400 and i < 4421) or (i > 5100 and i < 5216):
            if int(yTest[i][0])==0:
                index_of_0s.append(i)
            elif int(yTest[i][0])==4:
                index_of_4s.append(i)
            elif int(yTest[i][0])==2:
                index_of_2s.append(i)
            if len(index_of_0s) > 5 and \
               len(index_of_4s) > 5 and len(index_of_2s) > 5:
                break
    print("size 2s",len(index_of_2s),"size 0s",len(index_of_0s),"size 4s",len(index_of_4s))
    # -- Use target dataset, y, to print the data pointed to
    # -- by the collected indexes
    for index in index_of_0s:
        print('y at index',index, 'is',yTest[index])
    for index in index_of_4s:
        print('y at index',index, 'is',yTest[index])
    for index in index_of_2s:
        print('y at index',index, 'is',yTest[index])
    indexof2024=[]
    indexof2024.append(index_of_2s[3])
    indexof2024.append(index_of_0s[3])
    indexof2024.append(index_of_2s[4])
    indexof2024.append(index_of_4s[5])
    '''
    print("Index of 2024 ",indexof2024)
    horizontal_plot(XTest,yTest,indexof2024,4)
    '''
    # -- Plot your data. Use 6 characters per row ---
    '''
    horizontal_plot(XTest,yTest,index_of_2s[2],1)
    horizontal_plot(XTtest,yTest,index_of_0s[3],1)
    horizontal_plot(XTest,yTest,index_of_2s[4],1)
    horizontal_plot(XTest,yTest,index_of_4s[5],1)
    #horizontal_plot(XTrain,yTrain,index_of_8s,6)
    '''
  
# -- Processing starts here
if __name__ == '__main__':
    main()
 
