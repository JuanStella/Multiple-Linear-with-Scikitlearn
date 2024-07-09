from pydoc import doc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class Algorithm:

    def __init__(self):
        self.x = None
        self.y = None
        self.data = None

    def read_data(self, file):
        self.data = np.loadtxt(file, delimiter=',', skiprows=1)
        return None

    def algorithm(self, data):
        
        self.x = np.zeros((len(data), 2))
        self.x[:, 0] = data[:, 0]
        self.x[:, 1] = data[:, 1]

        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        self.y = data[:, 2]
        self.y = np.array(self.y)

        sgdr = SGDRegressor(max_iter=1000)
        sgdr.fit(self.x, self.y)

        b_norm = sgdr.intercept_
        w_norm = sgdr.coef_

        # make a prediction using w,b. 
        y_pred = np.dot(self.x, w_norm) + b_norm
        X_features = ['age','experience']

        # plot predictions and targets vs original features    
        fig,ax=plt.subplots(1,2,figsize=(12,3),sharey=True)
        for i in range(len(ax)):
            ax[i].scatter(self.x[:,i],self.y, label = 'target')
            ax[i].set_xlabel(X_features[i])
            ax[i].scatter(self.x[:,i],y_pred,color="red", label = 'predict')
        ax[0].set_ylabel("Price"); ax[0].legend();
        fig.suptitle("target versus prediction using z-score normalized model")
        plt.show()
        
        return y_pred, scaler
    

def main():
    file = 'D:\\ML\\MyProgress\\Multiple Linear with Scikitlearn\\archive\\data.csv'

    # Create an object of the class Algorithm
    model = Algorithm()
    model.read_data(file)
    ypred, scaler = model.algorithm(model.data)
    
    model.x = scaler.inverse_transform(model.x)
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model.x[:, 0], model.x[:, 1], model.y, color='b', label='Data Points')
    ax.plot_trisurf(model.x[:, 0], model.x[:, 1], ypred, color='r', alpha=0.5, label='Regression Plane')
    ax.set_xlabel('Age')
    ax.set_ylabel('Yrs of Exp')
    ax.set_zlabel('Income')
    ax.legend()
    plt.show()


    return None


if __name__ == '__main__':
    main()