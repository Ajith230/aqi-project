import os

import joblib

from Demo import settings


def predict_linear_regression(so2, no2, rspm, spm):

    #LR = joblib.load('linear_regression.joblib')
    LR = joblib.load((os.path.join(settings.BASE_DIR, 'linear_regression.joblib')))
    # predictionrslt = LR.predict([[4.8, 21.75, 78.18, 102]])
    predictionrslt = LR.predict([[so2, no2, rspm, spm]])
    return predictionrslt[0]
    # print('R^2_Square:%.2f ' % r2_score(y_test, predictions))
    # print('MSE:%.2f ' % np.sqrt(mean_squared_error(y_test, predictions)))


def predict_SVModel(so2, no2, rspm, spm):
    srv_model = joblib.load('svmodel.joblib')
    predictionrslt = srv_model.predict([[so2, no2, rspm, spm]])
    return predictionrslt[0]

def predict_logistic_regression(so2, no2, rspm, spm):
    model1 = joblib.load('logisticregression.joblib')
    predictionrslt = model1.predict([[so2, no2, rspm, spm]])
    return predictionrslt[0]
def predict_random_classifier(so2, no2, rspm, spm):
    model1 = joblib.load('randomclassifier.joblib')
    predictionrslt = model1.predict([[so2, no2, rspm, spm]])
    return predictionrslt[0]
def predict_random_regressor(so2, no2, rspm, spm):
    model1 = joblib.load('randomregressor.jobib')
    predictionrslt = model1.predict([[so2, no2, rspm, spm]])
    return predictionrslt[0]

#print("with linear regression ", predict_linear_regression(so2=4.8, no2=21.75, rspm=78.18, spm=102))
#print("with SVMODEL ", predict_SVModel(so2=4.8, no2=21.75, rspm=78.18, spm=102))
#print("with logistic regression ", predict_logistic_regression(so2=4.8, no2=21.75, rspm=78.18, spm=102))
#print("with random classifier ", predict_random_classifier(so2=4.8, no2=21.75, rspm=78.18, spm=102))
#print("with random regressor ", predict_random_regressor(so2=4.8, no2=21.75, rspm=78.18, spm=102))
#print( predict_SVModel(so2=5.2, no2=6.1, rspm=76.53653, spm=75.0))
