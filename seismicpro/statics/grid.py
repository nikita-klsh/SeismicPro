from .dataset import TravelTimeDataset
from ..const import HDR_FIRST_BREAK


class Grid:
    dataset_class = TravelTimeDataset

    def __init__(self, survey=None):
        self.survey = survey

    @property
    def has_survey(self):
        return self.survey is not None

    # Dataset generation

    def create_dataset(self, survey=None, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                       **kwargs):
        if survey is None:
            if not self.has_survey:
                raise ValueError("A survey to create a dataset must be passed")
            survey = self.survey
        return self.dataset_class(survey, self, first_breaks_header, uphole_correction_method, **kwargs)
