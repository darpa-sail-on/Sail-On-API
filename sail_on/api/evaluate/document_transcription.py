"""Document Transcription Class for metrics for sail-on."""

from .metrics import ProgramMetrics

import numpy as np
from pandas import DataFrame

from typing import Dict


class DocumentTranscriptionMetrics(ProgramMetrics):
    """Document transcription program metric class."""

    def __init__(
        self,
        protocol: str,
        image_id: int,
        text: int,
        novel: int,
        representation: int,
        detection: int,
        classification: int,
        pen_pressure: int,
        letter_size: int,
        word_spacing: int,
        slant_angle: int,
        attribute: int,
    ) -> None:
        """
        Initialize.
        Args:
            protocol: Name of the protocol.
            image_id: Column id for image
            text: Transcription associated with the image
            novel: Column id for predicting if change was detected
            representation: Column id with representation novelty label
            detection: Column id with sample wise novelty
            classification:  Column id with writer id
            pen_pressure: Column id with pen pressure values
            letter_size: Column id with letter size values
            word_spacing: Column id with word spacing values
            slant_angle: Column id with slant angle values
            attribute: Column id with attribute level novelty label
        Returns:
            None
        """
        super().__init__(protocol)
        self.image_id = image_id
        self.text_id = text
        self.novel_id = novel
        self.representation_id = representation
        self.detection_id = detection
        self.classification_id = classification
        self.pen_pressure_id = pen_pressure
        self.letter_size_id = letter_size
        self.word_spacing_id = word_spacing
        self.slant_angle_id = slant_angle
        self.attribute_id = attribute

    def m_acc(
        self,
        gt_novel: DataFrame,
        p_class: DataFrame,
        gt_class: DataFrame,
        round_size: int,
        asymptotic_start_round: int,
    ) -> Dict:
        """
        m_acc helper function used for computing novelty reaction performance.
        Args:
            gt_novel: ground truth detections (Dimension: [img X detection])
            p_class: detection predictions (Dimension: [img X prob that sample is novel, prob of known classes])
            gt_class: ground truth classes (Dimension: [img X class idx])
            round_size: size of the round
            asymptotic_start_round: asymptotic samples considered for computing metrics
        Returns:
            Dictionary containing top1, top3 accuracy over the test, pre and post novelty.
        """
        from evaluate.metrics import M_acc

        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_acc(
            gt_novel, class_prob, gt_class_idx, round_size, asymptotic_start_round
        )

    def m_num(self, p_novel: DataFrame, gt_novel: DataFrame) -> float:
        """
        m_num function.
        A Program Metric where the number of samples needed for detecting novelty.
        The method computes the number of GT novel samples needed to predict the
        first true positive.
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Difference between the novelty introduction and predicting change in world.
        """
        from evaluate.metrics import M_num
        return M_num(p_novel, gt_novel)

    def m_num_stats(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_num_stats function.
        Number of samples needed for detecting novelty. The method computes
        number of GT novel samples needed to predict the first true positive.
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Dictionary containing indices for novelty introduction and change in world prediction.
        """
        from evaluate.metrics import M_num_stats
        return M_num_stats(p_novel, gt_novel)

    def m_ndp(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp function.
        Novelty detection performance. The method computes per-sample novelty
        detection performance over the entire test.
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Dictionary containing novelty detection performance over the test.
        """
        from evaluate.metrics import M_ndp
        return M_ndp(p_novel, gt_novel)

    def m_ndp_pre(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_pre function.
        See :func:`~sail-on-client.evaluate.transcription.DocumentTranscriptionMetrics.m_ndp`
        with post_novelty. This computes to the first GT novel sample. It really isn't useful
        and is just added for completion. Should always be 0 since no possible TP.
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Dictionary containing detection performance pre novelty.
        """
        from evaluate.metrics import M_ndp
        return M_ndp(p_novel, gt_novel, mode="pre_novelty")

    def m_ndp_post(self, p_novel: np.ndarray, gt_novel: np.ndarray) -> Dict:
        """
        m_ndp_post function.
        See :func:`~sail-on-client.evaluate.transcription.DocumentTranscriptionMetrics.m_ndp`
        with post_novelty. This computes from the first GT novel sample.
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Dictionary containing detection performance post novelty.
        """
        from evaluate.metrics import M_ndp
        return M_ndp(p_novel, gt_novel, mode="post_novelty")

    def m_ndp_failed_reaction(
        self,
        p_novel: DataFrame,
        gt_novel: DataFrame,
        p_class: DataFrame,
        gt_class: DataFrame,
    ) -> Dict:
        """
        m_ndp_failed_reaction function.
        Not Implemented since no gt_class info for novel samples. The method
        computes novelty detection performance for only on samples with
        incorrect k-class predictions
        Args:
            p_novel: detection predictions (Dimension: [img X novel])
            gt_novel: ground truth detections (Dimension: [img X detection])
            p_class: detection predictions (Dimension: [img X prob that sample is novel, prob of known classes])
            gt_class: ground truth classes (Dimension: [img X class idx])
        Returns:
            Dictionary containing TP, FP, TN, FN, top1, top3 accuracy over the test.
        """
        from evaluate.metrics import M_ndp_failed_reaction

        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_ndp_failed_reaction(p_novel, gt_novel, class_prob, gt_class_idx)

    def m_accuracy_on_novel(
        self, p_class: DataFrame, gt_class: DataFrame, gt_novel: DataFrame
    ) -> Dict:
        """
        Additional Metric: Novelty robustness.
        Not Implemented since no gt_class info for novel samples. The method
        computes top-K accuracy for only the novel samples
        Args:
            p_class: detection predictions (Dimension: [img X prob that sample is novel, prob of known classes])
            gt_class: ground truth classes (Dimension: [img X class idx])
            gt_novel: ground truth detections (Dimension: [img X detection])
        Returns:
            Accuracy on novely samples
        """
        from evaluate.metrics import M_accuracy_on_novel

        class_prob = p_class.iloc[:, range(1, p_class.shape[1])].to_numpy()
        gt_class_idx = gt_class.to_numpy()
        return M_accuracy_on_novel(class_prob, gt_class_idx, gt_novel)

    def m_is_cdt_and_is_early(self, gt_idx: int, ta2_idx: int, test_len: int) -> Dict:
        """
        Is change detection and is change detection early (m_is_cdt_and_is_early) function.
        Args:
            gt_idx: Index when novelty is introduced
            ta2_idx: Index when change is detected
            test_len: Length of test
        Returns
            Dictionary containing boolean showing if change was was detected and if it was detected early
        """
        is_cdt = bool((ta2_idx >= gt_idx) & (ta2_idx < test_len))
        is_early = bool(ta2_idx < gt_idx)
        return {"Is CDT": is_cdt, "Is Early": is_early}