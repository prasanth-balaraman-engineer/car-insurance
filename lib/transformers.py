from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def get_time_difference(start_time, end_time, time_format, seconds=True):
    start_time = datetime.strptime(start_time, time_format)
    end_time = datetime.strptime(end_time, time_format)
    difference = end_time - start_time
    if seconds:
        return difference.total_seconds()
    else:
        return difference.total_seconds() / 60


class CallDurationInMinutesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, format_string='%H:%M:%S', call_start_column='CallStart', call_end_column='CallEnd',
                 output_column='CallDurationMins'):
        self.format_string = format_string
        self.call_start_column = call_start_column
        self.call_end_column = call_end_column
        self.output_column = output_column

    def fit(self, *args, **kwargs):
        return self

    def get_time_in_minutes(self, row):
        return get_time_difference(row[self.call_start_column], row[self.call_end_column], self.format_string,
                                   seconds=False)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.output_column] = output_df.apply(self.get_time_in_minutes, axis=1).astype(int)
        return output_df


class CallDurationInSecondsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, format_string='%H:%M:%S', call_start_column='CallStart', call_end_column='CallEnd',
                 output_column='CallDurationSecs'):
        self.format_string = format_string
        self.call_start_column = call_start_column
        self.call_end_column = call_end_column
        self.output_column = output_column

    def fit(self, *args, **kwargs):
        return self

    def get_time_in_seconds(self, row):
        return get_time_difference(row[self.call_start_column], row[self.call_end_column], self.format_string,
                                   seconds=True)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.output_column] = output_df.apply(self.get_time_in_seconds, axis=1).astype(int)
        return output_df


class OrdinalMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_column=None, mapping_dict=None):
        self.feature_column = feature_column
        self.mapping_dict = mapping_dict

    def fit(self, *args, **kwargs):
        return self

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.feature_column] = (output_df[self.feature_column]
                                          .map(self.mapping_dict)
                                          .fillna(value=0)
                                          .astype('int', errors='ignore'))
        return output_df


class SimpleImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_column, imputation_params_dict):
        self.feature_column = feature_column
        self.imputation_params_dict = imputation_params_dict

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, input_df):
        output_df = input_df.copy()
        params = self.imputation_params_dict
        output_df[self.feature_column] = SimpleImputer(**params).fit_transform(
            output_df[[self.feature_column]])
        return output_df


class SelectFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features

    def fit(self, *args, **kwargs):
        return self

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df = output_df[self.features]
        return output_df
