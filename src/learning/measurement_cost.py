"""Shared candidate cost; none is represented separately by zero, never here."""
def candidate_communication_cost(link_risk, fixed_packet_fraction=0.5):
    if not 0.0 <= float(fixed_packet_fraction) <= 1.0:
        raise ValueError("fixed packet fraction must be in [0, 1]")
    return float(fixed_packet_fraction) + (1.0 - float(fixed_packet_fraction)) * link_risk
