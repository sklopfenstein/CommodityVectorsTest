# -*- coding: utf-8 -*-
from apistar import typesystem

COLS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'class'
]
PROPER_COLS_ORDER = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss",
    "hours-per-week", "workclass_ ?",
    "workclass_ Federal-gov",
    "workclass_ Local-gov", "workclass_ Never-worked",
    "workclass_ Private",
    "workclass_ Self-emp-inc", "workclass_ Self-emp-not-inc",
    "workclass_ State-gov", "workclass_ Without-pay",
    "education_ 10th", "education_ 11th", "education_ 12th",
    "education_ 1st-4th", "education_ 5th-6th",
    "education_ 7th-8th",
    "education_ 9th", "education_ Assoc-acdm",
    "education_ Assoc-voc", "education_ Bachelors",
    "education_ Doctorate", "education_ HS-grad",
    "education_ Masters", "education_ Preschool",
    "education_ Prof-school", "education_ Some-college",
    "marital-status_ Divorced",
    "marital-status_ Married-AF-spouse",
    "marital-status_ Married-civ-spouse",
    "marital-status_ Married-spouse-absent",
    "marital-status_ Never-married",
    "marital-status_ Separated",
    "marital-status_ Widowed", "occupation_ ?",
    "occupation_ Adm-clerical", "occupation_ Armed-Forces",
    "occupation_ Craft-repair", "occupation_ Exec-managerial",
    "occupation_ Farming-fishing",
    "occupation_ Handlers-cleaners",
    "occupation_ Machine-op-inspct",
    "occupation_ Other-service",
    "occupation_ Priv-house-serv",
    "occupation_ Prof-specialty",
    "occupation_ Protective-serv", "occupation_ Sales",
    "occupation_ Tech-support",
    "occupation_ Transport-moving",
    "relationship_ Husband", "relationship_ Not-in-family",
    "relationship_ Other-relative", "relationship_ Own-child",
    "relationship_ Unmarried", "relationship_ Wife",
    "race_ Amer-Indian-Eskimo", "race_ Asian-Pac-Islander",
    "race_ Black", "race_ Other", "race_ White",
    "sex_ Female", "sex_ Male", "native-country_ ?",
    "native-country_ Cambodia", "native-country_ Canada",
    "native-country_ China", "native-country_ Columbia",
    "native-country_ Cuba",
    "native-country_ Dominican-Republic",
    "native-country_ Ecuador", "native-country_ El-Salvador",
    "native-country_ England", "native-country_ France",
    "native-country_ Germany", "native-country_ Greece",
    "native-country_ Guatemala", "native-country_ Haiti",
    "native-country_ Holand-Netherlands",
    "native-country_ Honduras", "native-country_ Hong",
    "native-country_ Hungary", "native-country_ India",
    "native-country_ Iran", "native-country_ Ireland",
    "native-country_ Italy", "native-country_ Jamaica",
    "native-country_ Japan", "native-country_ Laos",
    "native-country_ Mexico", "native-country_ Nicaragua",
    "native-country_ Outlying-US(Guam-USVI-etc)",
    "native-country_ Peru", "native-country_ Philippines",
    "native-country_ Poland", "native-country_ Portugal",
    "native-country_ Puerto-Rico", "native-country_ Scotland",
    "native-country_ South", "native-country_ Taiwan",
    "native-country_ Thailand",
    "native-country_ Trinadad&Tobago",
    "native-country_ United-States",
    "native-country_ Vietnam",
    "native-country_ Yugoslavia"
]
CATEGORICALS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
NON_CATEGORICALS = [col for col in COLS
                    if (col not in CATEGORICALS and col != 'class')]


# TODO: Move typesystem classes to a separate file.
class Age(typesystem.Integer):
    pass


class WorkClass(typesystem.Enum):
    enum = [
        'Private',
        'Self-emp-inc',
        'State-gov',
        'Local-gov',
        'Without-pay',
        'Self-emp-not-inc',
        'Federal-gov',
        'Never-worked',
        '?'
    ]


class Fnlwgt(typesystem.Number):
    pass


class Education(typesystem.Enum):
    enum = [
        '7th-8th',
        'Prof-school',
        '1st-4th',
        'Assoc-voc',
        'Masters',
        'Assoc-acdm',
        '9th',
        'Doctorate',
        'Bachelors',
        '5th-6th',
        'Some-college',
        '10th',
        '11th',
        'HS-grad',
        'Preschool',
        '12th'
    ]


class EducationNum(typesystem.Integer):
    pass


class MaritalStatus(typesystem.Enum):
    enum = [
        'Separated',
        'Divorced',
        'Married-spouse-absent',
        'Widowed',
        'Married-AF-spouse',
        'Never-married',
        'Married-civ-spouse'
    ]


class Occupation(typesystem.Enum):
    enum = [
        'Armed-Forces',
        'Craft-repair',
        'Other-service',
        'Transport-moving',
        'Prof-specialty',
        'Sales',
        'Machine-op-inspct',
        'Exec-managerial',
        'Handlers-cleaners',
        'Protective-serv',
        'Adm-clerical',
        'Tech-support',
        '?',
        'Farming-fishing',
        'Priv-house-serv'
    ]


class Relationship(typesystem.Enum):
    enum = [
        'Wife',
        'Own-child',
        'Unmarried',
        'Husband',
        'Other-relative',
        'Not-in-family'
    ]


class Race(typesystem.Enum):
    enum = [
        'Asian-Pac-Islander',
        'White',
        'Other',
        'Amer-Indian-Eskimo',
        'Black'
    ]


class Sex(typesystem.Enum):
    enum = [
        'Male',
        'Female'
    ]


class CapitalGain(typesystem.Number):
    pass


class CapitalLoss(typesystem.Number):
    pass


class HoursPerWeek(typesystem.Integer):
    pass


class NativeCountry(typesystem.Enum):
    enum = [
        'Iran',
        'Cuba',
        'Puerto-Rico',
        'Outlying-US(Guam-USVI-etc)',
        'El-Salvador',
        'Guatemala',
        'Holand-Netherlands',
        'United-States',
        'China',
        'Thailand',
        'Haiti',
        'Germany',
        'Columbia',
        'Hungary',
        'Dominican-Republic',
        'Poland',
        'Philippines',
        'Trinadad&Tobago',
        'Vietnam',
        'South',
        'Honduras',
        'Mexico',
        'Portugal',
        'England',
        'Jamaica',
        'India',
        'Yugoslavia',
        'Greece',
        'Japan',
        'Taiwan',
        '?',
        'Nicaragua',
        'Canada',
        'Hong',
        'Italy',
        'Scotland',
        'France',
        'Cambodia',
        'Ecuador',
        'Laos',
        'Peru',
        'Ireland'
    ]

 
class SalaryClass(typesystem.Enum):
    enum = [' >50K', ' >50K.', ' <=50K', ' <=50K.']


class Person(typesystem.Object):
    properties = {
        'age': Age,
        'workclass': WorkClass,
        'fnlwgt': Fnlwgt,
        'education': Education,
        'education-num': EducationNum,
        'marital-status': MaritalStatus,
        'occupation': Occupation,
        'relationship': Relationship,
        'race': Race,
        'sex': Sex,
        'capital-gain': CapitalGain,
        'capital-loss': CapitalLoss,
        'hours-per-week': HoursPerWeek,
        'native-country': NativeCountry
    }


class ClassifiedPerson(Person):
    def __init__(self):
        self.properties['class'] = SalaryClass


class Sample(typesystem.Array):
    items = Person
    description = """A Sample is a JSON array containing several JSON objects with fields:\n
        'age': positive integer \\ 
        'workclass': handled categories are 'Private', 'Self-emp-inc', 'State-gov', 'Local-gov', 'Without-pay', 'Self-emp-not-inc', 'Federal-gov', 'Never-worked' or '?' \\ 
        'fnlwgt': positive integer \\ 
        'education': handled categories are '7th-8th', 'Prof-school', '1st-4th', 'Assoc-voc', 'Masters', 'Assoc-acdm', '9th', 'Doctorate', 'Bachelors', '5th-6th', 'Some-college', '10th', '11th', 'HS-grad', 'Preschool' or '12th' \\ 
        'education-num': positive integer \\ 
        'marital-status': handled categories are 'Separated', 'Divorced', 'Married-spouse-absent', 'Widowed', 'Married-AF-spouse', 'Never-married', 'Married-civ-spouse' \\ 
        'occupation': handled categories are 'Armed-Forces', 'Craft-repair', 'Other-service', 'Transport-moving', 'Prof-specialty', 'Sales', 'Machine-op-inspct', 'Exec-managerial', 'Handlers-cleaners', 'Protective-serv', 'Adm-clerical', 'Tech-support', '?', 'Farming-fishing', 'Priv-house-serv' \\ 
        'relationship': handled categories are 'Wife', 'Own-child', 'Unmarried', 'Husband', 'Other-relative', 'Not-in-family' \\ 
        'race': handled categories are 'Asian-Pac-Islander', 'White', 'Other', 'Amer-Indian-Eskimo', 'Black' \\ 
        'sex': handled categories are 'Male', 'Female' \\ 
        'capital-gain': positive integer \\ 
        'capital-loss': positive integer \\ 
        'hours-per-week': positive integer \\ 
        'native-country': handled categories are 'Iran', 'Cuba', 'Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'El-Salvador', 'Guatemala', 'Holand-Netherlands', 'United-States', 'China', 'Thailand', 'Haiti', 'Germany', 'Columbia', 'Hungary', 'Dominican-Republic', 'Poland', 'Philippines', 'Trinadad&Tobago', 'Vietnam', 'South', 'Honduras', 'Mexico', 'Portugal', 'England', 'Jamaica', 'India', 'Yugoslavia', 'Greece', 'Japan', 'Taiwan', '?', 'Nicaragua', 'Canada', 'Hong', 'Italy', 'Scotland', 'France', 'Cambodia', 'Ecuador', 'Laos', 'Peru', 'Ireland' \\ 

    """  # noqa


class ClassifiedSample(typesystem.Array):
    items = ClassifiedPerson
    description = """A ClassifiedSample is a JSON array containing several JSON objects with fields:\n
        'age': positive integer \\ 
        'workclass': handled categories are 'Private', 'Self-emp-inc', 'State-gov', 'Local-gov', 'Without-pay', 'Self-emp-not-inc', 'Federal-gov', 'Never-worked' or '?' \\ 
        'fnlwgt': positive integer \\ 
        'education': handled categories are '7th-8th', 'Prof-school', '1st-4th', 'Assoc-voc', 'Masters', 'Assoc-acdm', '9th', 'Doctorate', 'Bachelors', '5th-6th', 'Some-college', '10th', '11th', 'HS-grad', 'Preschool' or '12th' \\ 
        'education-num': positive integer \\ 
        'marital-status': handled categories are 'Separated', 'Divorced', 'Married-spouse-absent', 'Widowed', 'Married-AF-spouse', 'Never-married', 'Married-civ-spouse' \\ 
        'occupation': handled categories are 'Armed-Forces', 'Craft-repair', 'Other-service', 'Transport-moving', 'Prof-specialty', 'Sales', 'Machine-op-inspct', 'Exec-managerial', 'Handlers-cleaners', 'Protective-serv', 'Adm-clerical', 'Tech-support', '?', 'Farming-fishing', 'Priv-house-serv' \\ 
        'relationship': handled categories are 'Wife', 'Own-child', 'Unmarried', 'Husband', 'Other-relative', 'Not-in-family' \\ 
        'race': handled categories are 'Asian-Pac-Islander', 'White', 'Other', 'Amer-Indian-Eskimo', 'Black' \\ 
        'sex': handled categories are 'Male', 'Female' \\ 
        'capital-gain': positive integer \\ 
        'capital-loss': positive integer \\ 
        'hours-per-week': positive integer \\ 
        'native-country': handled categories are 'Iran', 'Cuba', 'Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'El-Salvador', 'Guatemala', 'Holand-Netherlands', 'United-States', 'China', 'Thailand', 'Haiti', 'Germany', 'Columbia', 'Hungary', 'Dominican-Republic', 'Poland', 'Philippines', 'Trinadad&Tobago', 'Vietnam', 'South', 'Honduras', 'Mexico', 'Portugal', 'England', 'Jamaica', 'India', 'Yugoslavia', 'Greece', 'Japan', 'Taiwan', '?', 'Nicaragua', 'Canada', 'Hong', 'Italy', 'Scotland', 'France', 'Cambodia', 'Ecuador', 'Laos', 'Peru', 'Ireland' \\ 
        'class': handled classes are '>50K' and '<=50K'. '>50K.' and '<=50K.' are tolerated as well \\ 

    """  # noqa


class Page(typesystem.Integer):
    minimum = 0
    description = """Positive Integer. Data is paginated. The index of the page desired."""  # noqa


class Size(typesystem.Integer):
    minimum = 0
    maximum = 50
    description = """Positive Integer. Page size. Must be inferior to 50."""
