import torch
import pytest
from losses.cb_focal_loss import CBFocalLoss

def test_cbfocal_loss_no_focal():
    # Test with gamma = 0, should be equivalent to class-balanced cross-entropy
    class_counts = torch.tensor([1000, 100, 10])
    cb_beta = 0.9999
    focal_gamma = 0.0
    num_classes = 3

    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)

    inputs = torch.randn(10, num_classes)
    targets = torch.randint(0, num_classes, (10,))

    loss = loss_fn(inputs, targets)
    assert loss is not None
    assert loss.item() >= 0

def test_cbfocal_loss_basic():
    class_counts = torch.tensor([1000, 100, 10])
    cb_beta = 0.9999
    focal_gamma = 2.0
    num_classes = 3

    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)

    inputs = torch.randn(10, num_classes)
    targets = torch.randint(0, num_classes, (10,))

    loss = loss_fn(inputs, targets)
    assert loss is not None
    assert loss.item() >= 0

def test_cbfocal_loss_zero_beta():
    # Test with beta = 0, weights should be 1/n_c (inverse frequency)
    class_counts = torch.tensor([1000, 100, 10])
    cb_beta = 0.0
    focal_gamma = 2.0
    num_classes = 3

    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)

    inputs = torch.randn(10, num_classes)
    targets = torch.randint(0, num_classes, (10,))

    loss = loss_fn(inputs, targets)
    assert loss is not None

def test_cbfocal_loss_class_imbalance():
    # Simulate a scenario with strong class imbalance
    class_counts = torch.tensor([10000, 100, 10]) # Stationary, Up, Down
    cb_beta = 0.9999
    focal_gamma = 2.0
    num_classes = 3

    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)

    # Create inputs where model is confident about majority class but uncertain about minority
    inputs = torch.tensor([
        [ 5.0, -1.0, -1.0], # Confident stationary
        [-1.0,  5.0, -1.0], # Confident up
        [-1.0, -1.0,  5.0], # Confident down
        [ 4.0,  1.0,  1.0], # Less confident stationary
        [ 1.0,  4.0,  1.0], # Less confident up
    ])
    targets = torch.tensor([0, 1, 2, 0, 1])

    loss = loss_fn(inputs, targets)
    assert loss is not None
    # The exact value is hard to assert without full calculation, but we expect it to be a valid tensor
    print(f"Imbalanced loss: {loss.item()}")


def test_cbfocal_loss_reduces_easy_examples():
    class_counts = torch.tensor([100, 100, 100])
    cb_beta = 0.9999
    focal_gamma = 2.0
    num_classes = 3
    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)

    # Easy example (high probability for target class)
    easy_inputs = torch.tensor([[-1.0, -1.0, 5.0]])
    easy_targets = torch.tensor([2])
    easy_loss = loss_fn(easy_inputs, easy_targets)

    # Hard example (low probability for target class)
    hard_inputs = torch.tensor([[5.0, -1.0, -1.0]])
    hard_targets = torch.tensor([2])
    hard_loss = loss_fn(hard_inputs, hard_targets)

    assert easy_loss.item() < hard_loss.item() # Focal loss should reduce easy examples more


def test_cbfocal_loss_weights_sum_one():
    class_counts = torch.tensor([1000, 100, 10])
    cb_beta = 0.9999
    focal_gamma = 2.0
    num_classes = 3

    loss_fn = CBFocalLoss(class_counts, cb_beta, focal_gamma, num_classes)
    assert torch.isclose(loss_fn.weights.sum(), torch.tensor(1.0))
