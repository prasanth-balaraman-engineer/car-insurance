NUMERIC_COLUMNS = ['LastContactDay', 'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'CallDurationMins',
                   'CallDurationSecs']
SCALABLE_NUMERIC_COLUMNS = ['Age', 'Balance']
ONE_HOT_ENCODING_COLUMNS = ['Job', 'Marital', 'Communication', 'LastContactMonth']
BINARY_COLUMNS = ['Default', 'HHInsurance', 'CarLoan', 'Outcome']
ALL_COLUMNS = NUMERIC_COLUMNS + SCALABLE_NUMERIC_COLUMNS + ONE_HOT_ENCODING_COLUMNS + BINARY_COLUMNS

EDUCATION_MAPPING = dict(primary=1, secondary=2, tertiary=3)
OUTCOME_MAPPING = dict(failure=0, other=0, success=1)

JOB_IMPUTATION_PARAMS_DICT = dict(strategy='constant', fill_value='unknown')
COMMUNICATION_IMPUTATION_PARAMS_DICT = dict(strategy='constant', fill_value='unknown')
