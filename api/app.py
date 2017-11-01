# coding: utf-8

from apistar import Include, Route, typesystem
from apistar.frameworks.asyncio import ASyncIOApp as App
from apistar.handlers import docs_urls, static_urls
from hashlib import sha1
from json import dumps
import multiprocessing
from multiprocessing import Manager
import os
import pandas as pd
import pickle

from predictor import classify
from schemas import COLS, Page, Sample, Size
from settings import CONSUMERS_NUMBER, DATASET_RELATIVE_LOCATION


# TODO: use XGBoost classifier

# TODO: write tests (APIstar Mixins?).

# TODO: Security? (APIstar presets?)

# TODO: create requirements.txt.

# TODO: Create docker container.

# TODO: Move static variables to settings.py when meaningful.

# TODO: Write tests

cols = COLS[:-1]

dir_path = os.path.abspath(os.getcwd()) + '/'
prediction_jobqueue = multiprocessing.JoinableQueue()
completed_jobs = Manager().dict()


def classify_data(sample: Sample) -> dict:
    """Async function to add a new job to the prediction jobqueue.

    Args:
        sample (Sample): the array of objects to classify.

    """

    # The prediction task can be very long if the sample contains
    # a lot of rows to predict from.
    # This could lead to some performance issues if everything is
    # done synchronously (if every request to the '/predict' route
    # leads to a new prediction being synchronously performed).
    # It can make the CPU and memory consumption skyrocket.

    # We propose an architectural solution to solve this issue.

    # The idea is that every request to the '/predict' route will
    # create a new job in the predition jobs queue.
    # On the other side, a limited number of consumers will
    # consume these jobs in parallel.

    # By limiting the number of consumers to (for example) 4, we limit
    # the maximal amount of parallel computations.
    # This amount is no longer coupled with the number of requests,
    # making the system more robust.

    # We could still enhance this architecture to better meet REST
    # best practices for async/threaded computations.
    # We could first insert the job in the queue, then send a 303 redirect to
    # an async method that collects the computation result.
    # Currently with APIstar, redirection Responses seem a bit buggy.

    # For the sake of simplicity, this function is not using
    # python3 async capabilities.
    # If the 'while true' condition becomes a performance issue,
    # this should be changed to an async implementation.

    id = sha1()
    id.update(dumps(sample).encode('utf8'))
    id = id.hexdigest()

    # if classifications for this sample already exist, do not predict
    # but return existing result instead.
    if id in completed_jobs:
        return completed_jobs[id]

    job = {
        'id': id,
        'sample': sample
    }
    prediction_jobqueue.put(job)

    while True:
        if id in completed_jobs:
            return completed_jobs[id]


def format_strings(sample: list) -> list:
    """Because of the way we parsed Data to train the classifier,
    it is important to put a space at the begining of each string
    in the data for classification.

    TODO: Fix this.

    Args:
        sample (list): a list of dicts

    """
    new_sample = []

    for el in sample:
        for key, val in el.items():
            if type(val) is str:
                el[key] = ' ' + val
        new_sample.append(el)

    return new_sample


def consume_prediction_job(job: dict) -> (str, dict):
    """The sync function for workers to classify samples from the job queue.

    A worker pops a job from the queue, and classifies the associated sample.
    Then, it sends the result back to the caller with status 200.

    TODO: Handle failures (status code 305 when server error and 410 when job
    not in queue anymore after program restart).

    Args:
        job (dict): contains keys id, request and sample.

    Returns:
       Response with code 200, 305 or 410.

    """
    formated_sample = format_strings(job['sample'])
    sample_as_df = pd.DataFrame(formated_sample)
    prediction = classify(sample_as_df)
    pred_as_df = pd.DataFrame(prediction)
    pred_as_df.columns = ['class']
    full_profiles = pd.concat([sample_as_df, pred_as_df],
                              axis=1)
    return job['id'], list(full_profiles.T.to_dict().values())


def worker():
    """The daemon worker part for background parallel prediction execution.

    This is only meant to be ran in backround daemon process.

    """
    while True:
        job = prediction_jobqueue.get()
        id, prediction = consume_prediction_job(job)
        prediction_jobqueue.task_done()
        completed_jobs[id] = prediction


procs = []

for i in range(CONSUMERS_NUMBER):
    procs.append(multiprocessing.Process(target=worker))
    procs[-1].daemon = True
    procs[-1].start()


def format_classes(output_classes: pd.DataFrame) -> pd.DataFrame:
    """Formats the predictor's Boolean output classes to human readable classes.

    Args:
        output_classes (pandas.DataFrame): the binary classes asample (0|1).

    Returns:
        pandas.DataFrame of human readable classes ('>50K'|'<=50K').

    """
    return output_classes.replace(1, '>50K').replace(0, '<=50K')


def get_test_results() -> dict:
    """Get metrics accuracy, precision, recall for the evaluation
    on the test dataset.

    Returns:
        dict.

    """
    with open(dir_path + DATASET_RELATIVE_LOCATION + 'test_results.pkl',
              'rb') as f:
        return pickle.load(f)


def get_classified_test_set(page: Page,
                            size: Size) -> dict:
    """Get classifier output for the prediction on the test dataset.

    The data is paginated with size elements per page.
    One can ask the page that he needs.

    Max page size is 50.


    Args:
        page (typesystem.Integer): the page to get. pages have size 'size'.
        size (typesystem.Integer): the size of a page.

    Returns:
        dict.

    """
    # TODO: change this method to use the same structure as classify_data,
    # using the queue and the daemons.
    try:
        df = pd.read_csv(dir_path + DATASET_RELATIVE_LOCATION +
                         'classifier_output.csv',
                         skiprows=page * size, nrows=page * size + size,
                         header=None)
    except pd.errors.EmptyDataError:
        return {'message': 'end reached'}
    df.columns = ['index', 'age', 'workclass', 'fnlwgt', 'education',
                  'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country',
                  'ground_truth', 'classifier output']
    return list(df.T.to_dict().values())


# All the roads available in our API
routes = [
    Route('/classify/', 'POST', classify_data),
    Route('/test_results/', 'GET', get_test_results),
    Route('/test_set_classifier_output', 'GET', get_classified_test_set),

    # Below are autogenerated pages from APIstar.
    # Includes basic API documentation.
    # Beware: There are some test abilities on the docs page,
    # but these do not work when submiting json payload in the request body.
    # If you want to manually interact with the API, use Postman instead.
    Include('/docs', docs_urls),
    Include('/static', static_urls)
]

app = App(routes=routes)


if __name__ == '__main__':

    app.main()
