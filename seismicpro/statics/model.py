from .statics import Statics
from .utils import get_uphole_correction_method, group_source_headers, group_receiver_headers
from ..survey import Survey
from ..utils import to_list, align_args
from ..const import HDR_FIRST_BREAK


class NearSurfaceModel:
    def __init__(self, grid):
        self.grid = grid

    @classmethod
    def from_file(cls, path, **kwargs):
        raise NotImplementedError

    def dump(self, path, **kwargs):
        raise NotImplementedError

    def change_grid(self, grid, **kwargs):
        raise NotImplementedError

    # Dataset generation

    def create_dataset(self, survey=None, first_breaks_header=HDR_FIRST_BREAK, uphole_correction_method="auto",
                       **kwargs):
        raise NotImplementedError

    # Model fitting and inference

    def fit(self, dataset, batch_size, n_epochs, bar=True, **kwargs):
        raise NotImplementedError

    def predict(self, dataset, batch_size, bar=True, predicted_first_breaks_header=None):
        raise NotImplementedError

    # Statics calculation

    def calculate_source_statics(self, source_headers, source_id_cols, uphole_correction_method, **kwargs):
        raise NotImplementedError

    def calculate_receiver_statics(self, receiver_headers, receiver_id_cols, **kwargs):
        raise NotImplementedError

    def calculate_statics(self, survey=None, uphole_correction_method="auto", source_id_cols=None,
                          receiver_id_cols=None, **kwargs):
        if survey is None:
            if not self.grid.has_survey:
                raise ValueError("A survey to calculate statics for must be passed")
            survey = self.grid.survey
        survey_list = to_list(survey)
        is_single_survey = isinstance(survey, Survey)
        _, uphole_correction_method_list = align_args(survey_list, uphole_correction_method)

        if source_id_cols is None:
            if any(sur.source_id_cols != survey_list[0].source_id_cols for sur in survey_list):
                raise ValueError
            if survey_list[0].source_id_cols is None:
                raise ValueError
            source_id_cols = survey_list[0].source_id_cols

        if receiver_id_cols is None:
            if any(sur.receiver_id_cols != survey_list[0].receiver_id_cols for sur in survey_list):
                raise ValueError
            if survey_list[0].receiver_id_cols is None:
                raise ValueError
            receiver_id_cols = survey_list[0].receiver_id_cols

        source_statics_list = []
        receiver_statics_list = []
        for sur, uphole_correction_method in zip(survey_list, uphole_correction_method_list):
            source_headers = group_source_headers(sur, source_id_cols)
            uphole_correction_method = get_uphole_correction_method(sur, uphole_correction_method)
            source_statics = self.calculate_source_statics(source_headers, source_id_cols, uphole_correction_method,
                                                           **kwargs)
            source_statics_list.append(source_statics)

            receiver_headers = group_receiver_headers(sur, receiver_id_cols)
            receiver_statics = self.calculate_receiver_statics(receiver_headers, receiver_id_cols, **kwargs)
            receiver_statics_list.append(receiver_statics)

        survey = survey_list[0] if is_single_survey else survey_list
        source_statics = source_statics_list[0] if is_single_survey else source_statics_list
        receiver_statics = receiver_statics_list[0] if is_single_survey else receiver_statics_list
        return Statics(survey, source_statics, receiver_statics, source_id_cols=source_id_cols,
                       receiver_id_cols=receiver_id_cols)

    # Model visualization

    def plot_loss(self, show_reg=True, figsize=(10, 3)):
        raise NotImplementedError

    def plot_profile(self, **kwargs):
        raise NotImplementedError
