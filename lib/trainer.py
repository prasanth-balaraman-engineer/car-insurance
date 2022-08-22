from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


def run_classification_grid_search_for_single_classifier_type(classifier, params_grid, X, y, cv=3, verbose=0,
                                                              scoring='roc_auc'):
    grid = GridSearchCV(classifier, params_grid, cv=cv, n_jobs=-1, verbose=verbose, scoring=scoring)
    grid.fit(X, y)
    mean_score = grid.cv_results_['mean_test_score'][grid.best_index_]
    return grid.best_estimator_, mean_score, grid.best_params_


def run_classification_grid_search(params_grid, pipeline, X, y):
    best_score = 0
    best_classifier = None
    best_classifier_name = None
    best_params = None

    for name, model_params_grid in params_grid.items():
        model, grid = model_params_grid
        clf = make_pipeline(pipeline, model)

        print(f'Running grid search CV for {name} model')
        best_estimator, score, params = run_classification_grid_search_for_single_classifier_type(clf, grid, X, y)
        # print(f'Best estimator: {best_estimator}')
        # print(f'Best parameters: {params}')
        print(f'Best score: {score}\n')

        if score > best_score:
            best_score = score
            best_classifier = best_estimator
            best_classifier_name = name
            best_params = params

    return best_score, best_classifier, best_classifier_name, best_params
