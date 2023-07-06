import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import expand_dims
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ImageClassifier:
    def __init__(self, cat1_img, cat2_img, dog1_img, dog2_img):
        self.df = self.prepare_data(cat1_img, cat2_img, dog1_img, dog2_img)
        self.X_train_std, self.X_test_std, self.y_train, self.y_test = self.scaling()

    def prepare_data(self, cat1_img, cat2_img, dog1_img, dog2_img):
        cat1, cat2, dog1, dog2 = self.block_img(cat1_img), self.block_img(cat2_img), \
            self.block_img(dog1_img), self.block_img(dog2_img)

        df1, df2, df3, df4 = pd.DataFrame(cat1), pd.DataFrame(cat2), pd.DataFrame(dog1), pd.DataFrame(dog2)
        df1['label'], df2['label'] = 0, 0
        df3['label'], df4['label'] = 1, 1

        df = pd.concat([df1, df2, df3, df4], axis=0)
        return df

    def block_img(self, img_name):
        img = self.image_array(img_name)
        h, w = 8, 8
        feature = []
        for a in range(0, (img.shape[0] - h) + 1, h):
            for b in range(0, (img.shape[1] - w) + 1, w):
                block = img[a:a + h, b:b + w]
                my_block = (block.flatten())
                feature.append(my_block)
        return feature

    def image_array(self, name):
        img = load_img(name)
        img_array = img_to_array(img)
        return img_array

    def scaling(self):
        X = self.df.iloc[:, 0:192].values
        y = self.df.iloc[:, 192].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

        my_scaler = preprocessing.StandardScaler()

        X_train_std = my_scaler.fit_transform(X_train)
        X_test_std = my_scaler.transform(X_test)
        return X_train_std, X_test_std, y_train, y_test

    def lr_lda(self):
        lr_normal = LogisticRegression()
        lr_normal.fit(self.X_train_std, self.y_train)
        pred_normal = lr_normal.predict(self.X_test_std)
        print("Classification Report")
        print(classification_report(self.y_test, pred_normal))
        print("Confusion Report")
        print(confusion_matrix(self.y_test, pred_normal))
        return lr_normal, pred_normal

    def LDA(self):
        n_components = min(self.X_train_std.shape[1], len(np.unique(self.y_train))) - 1
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_LDA = lda.fit_transform(self.X_train_std, self.y_train)
        X_test_LDA = lda.transform(self.X_test_std)

        lr_lda = LogisticRegression()
        lr_lda.fit(X_train_LDA, self.y_train)

        pred_lda = lr_lda.predict(X_test_LDA)
        print("Classification Report")
        print(classification_report(self.y_test, pred_lda))
        print("Confusion Report")
        print(confusion_matrix(self.y_test, pred_lda))
        return pred_lda

    def LogisticRegression_PCA(self):
        pca = PCA(n_components=2)
        X_train_PCA = pca.fit_transform(self.X_train_std)
        X_test_PCA = pca.transform(self.X_test_std)

        log_r1 = LogisticRegression()
        log_r1.fit(X_train_PCA, self.y_train)

        pred1 = log_r1.predict(X_test_PCA)

        print("Classification Report")
        print(classification_report(self.y_test, pred1))

        print("Confusion Report")
        print(confusion_matrix(self.y_test, pred1))

        print(pca.explained_variance_ratio_)
        return pred1

    def PCA_LDA_Random(self):
        pipeline = Pipeline([('pca', PCA(n_components=2)),
                             ('clf', RandomForestRegressor(n_estimators=20, random_state=42))])
        pipeline.fit(self.X_train_std, self.y_train)
        print('Test Accuracy: %.3f' % pipeline.score(self.X_test_std, self.y_test))

        n_components = min(self.X_train_std.shape[1], len(np.unique(self.y_train))) - 1

        pipeline2 = Pipeline([('pca', LinearDiscriminantAnalysis(n_components=n_components)),
                              ('clf', RandomForestRegressor(n_estimators=20, random_state=42))])

        pipeline2.fit(self.X_train_std, self.y_train)
        print('Test Accuracy: %.3f' % pipeline2.score(self.X_test_std, self.y_test))

    def Sequential_Model(self):
        classifier = Sequential()
        classifier.add(Dense(units=800, kernel_initializer='uniform', activation='relu', input_shape=(192,)))
        classifier.add(Dense(units=600, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=400, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=2, kernel_initializer='uniform', activation='sigmoid'))

        optimizer = tf.keras.optimizers.SGD()
        classifier.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        classifier.fit(self.X_train_std, self.y_train, epochs=10, batch_size=32)

        predictions = classifier.predict(self.X_test_std)
        prediction = tf.argmax(predictions, axis=1)

        CC_test = tf.math.confusion_matrix(self.y_test, prediction)
        TN = CC_test[1, 1]
        FP = CC_test[1, 0]
        FN = CC_test[0, 1]
        TP = CC_test[0, 0]
        FPFN = FP + FN
        TPTN = TP + TN
        Accuracy = 1 / (1 + (FPFN / TPTN))
        print("Our_Accuracy_Score:", Accuracy)
        Precision = 1 / (1 + (FP / TP))
        print("Our_Precision_Score:", Precision)
        Sensitivity = 1 / (1 + (FN / TP))
        print("Our_Sensitivity_Score:", Sensitivity)
        Specificity = 1 / (1 + (FP / TN))
        print("Our_Specificity_Score:", Specificity)

        return prediction


# Usage Example
cat1_img = 'cat1.jpg'
cat2_img = 'cat2.jpg'
dog1_img = 'dog1.jpg'
dog2_img = 'dog2.jpg'

classifier = ImageClassifier(cat1_img, cat2_img, dog1_img, dog2_img)
classifier.lr_lda()
classifier.LDA()
classifier.LogisticRegression_PCA()
classifier.PCA_LDA_Random()
classifier.Sequential_Model()
