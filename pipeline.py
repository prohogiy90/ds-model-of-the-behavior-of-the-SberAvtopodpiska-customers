import os
import dill
import sklearn
import imblearn
import warnings
import datetime
import pandas as pd

from imblearn import FunctionSampler
from category_encoders import TargetEncoder
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier


path = os.environ.get('PROJECT_PATH', 'C:/Users/Konstantin/PycharmProjects/pythonProject/sberpodpiska_ml')


def features_add(data):
    import math
    import pandas as pnd
    # Creation of a new feature using the logarithmic function of the screen resolution
    data = data.copy()
    data.device_screen_resolution = data.device_screen_resolution.apply(lambda x: x.split('x'))
    data['device_screen_resolution_log'] = data.device_screen_resolution.apply(
        lambda x: round(math.log(int(x[0]) * int(x[1])), 2) if (int(x[0]) * int(x[1])) != 0 else 0)
    data['device_screen_resolution_log'] = \
        data['device_screen_resolution_log'].replace(0, data['device_screen_resolution_log'].median())
    # Converting Date and Time features
    data.visit_time = pnd.to_datetime(data.visit_time)
    data['hour'] = data.visit_time.dt.hour
    data.visit_date = pnd.to_datetime(data.visit_date)
    data['month'] = data.visit_date.dt.month
    data['dayofweek'] = data.visit_date.dt.weekday
    # Feature generation - the number of days from the start of the project to the date of the visit
    data['n_days_from_start'] = data.visit_date - data.visit_date.min()
    data['n_days_from_start'] = data['n_days_from_start'].astype(str).apply(lambda x: x.split(' ')[0]).astype('int64')
    # Converting the visit number with the logarithmic function
    data['visit_number_log'] = data.visit_number.apply(lambda x: math.log(x))
    # Generation of a new feature containing the name of the advertising company and the month of the visit
    data['utm_generic'] = data['utm_campaign'].astype(str) + '_' + data['month'].astype(str)
    return data


def drop_useless_columns(data):
    # Removing non-informative columns and original features before transformations
    columns_to_drop = [
        'visit_date',
        'visit_time',
        'visit_number',
        'utm_campaign',
        'device_screen_resolution',
        'month'
    ]
    return data.drop(columns_to_drop, axis=1)


def feature_sample(data, target):
    data = data.copy()
    target = target.copy()
    sempler = TomekLinks(sampling_strategy='auto', n_jobs=-1)
    data, target = sempler.fit_resample(data, target)
    return data, target


def main():
    warnings.simplefilter('ignore', UserWarning)

    df = pd.read_csv(f'{path}/model/data/sberpodpiska_data_prepared.csv')

    X = df.drop(['event_action_bin'], axis=1)
    y = df['event_action_bin']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_transform = ColumnTransformer(transformers=[
        ('numerical_transform', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical_transform', TargetEncoder(smoothing=0.5), make_column_selector(dtype_include=object))],
         n_jobs=-1)

    preprocessor = sklearn.pipeline.Pipeline(steps=[
        ('features_add', FunctionTransformer(features_add)),
        ('drop_columns', FunctionTransformer(drop_useless_columns)),
        ('transformation', feature_transform)
    ])

    models = (

        LGBMClassifier(boosting_type='dart', min_child_samples=60, n_estimators=300, n_jobs=-1, random_state=42,
                       learning_rate=0.08, is_unbalance=True),
        HistGradientBoostingClassifier(l2_regularization=30, max_iter=500, learning_rate=0.081, random_state=42),
        LogisticRegression(C=1E6, penalty='l2', solver='lbfgs', n_jobs=-1, max_iter=1000, class_weight={0: .5, 1: 13},
                           random_state=42),
        MLPClassifier(activation='relu', hidden_layer_sizes=(100,), max_iter=750, solver='adam', random_state=42),
        RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_split=8, n_jobs=-1, n_estimators=500,
                               class_weight='balanced', random_state=42),
        CatBoostClassifier(depth=10, learning_rate=0.04, iterations=500, task_type='GPU',
                           devices='0:1', gpu_ram_part=0.7, class_weights={0: .5, 1: 13}),
        StackingClassifier(estimators=[
            ('LGBM', LGBMClassifier(boosting_type='dart', min_child_samples=60, n_estimators=300, n_jobs=-1,
                                    learning_rate=0.08, random_state=42, is_unbalance=True)),
            ('HistGr', HistGradientBoostingClassifier(l2_regularization=30, max_iter=500, learning_rate=0.081,
                                                      random_state=42)),
            ('logreg', LogisticRegression(C=1E6, penalty='l2', solver='lbfgs', n_jobs=-1, max_iter=1000,
                                          class_weight={0: .5, 1: 13}, random_state=42))],
            final_estimator=RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_split=8, n_jobs=-1,
                                                   n_estimators=500, class_weight={0: .5, 1: 13}, random_state=42),
            n_jobs=-1)

    )

    best_score = .0
    best_pipe = None
    best_pred = []
    for model in models:
        pipe = sklearn.pipeline.Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('sample_classifier_pipeline', imblearn.pipeline.Pipeline(steps=[
                ('sampler', FunctionSampler(feature_sample)),
                ('classifier', model)
            ]))
        ])

        pipe.fit(train_x, train_y)
        proba = pipe.predict_proba(test_x)[:, 1]
        pred = pipe.predict(test_x)
        score = roc_auc_score(test_y, proba)
        print(f'model: {type(model).__name__}, roc_auc: {score:.4f}')
        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_pred = pred

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps.sample_classifier_pipeline.named_steps["classifier"]).__name__}, '
          f'roc_auc_score: {best_score:.4f}\n')
    print('Матрица ошибок:\n\n', confusion_matrix(test_y, best_pred), '\n')
    print('Значения метрик:\n', classification_report(test_y, best_pred))

    # Writing model to pkl-file
    with open('sberpodpiska.pkl', 'wb') as file:
        dill.dump({
           'model': best_pipe,
           'metadata': {
                'name': 'Sberpodpiska: customer behavior predicting',
                'author': 'Konstantin Skrebunov',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps.sample_classifier_pipeline.named_steps["classifier"]).__name__,
                'roc_auc_score': best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
