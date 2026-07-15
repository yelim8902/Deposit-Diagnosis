"""[D] 기대손실 = P(사고) × (보증금 − E[회수액])."""


def expected_loss(accident_probability, my_deposit_won, expected_recovery_won):
    loss_given_accident = max(my_deposit_won - expected_recovery_won, 0)
    return accident_probability * loss_given_accident
