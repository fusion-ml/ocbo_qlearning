def idx_to_state(state_idx, num_rows, num_cols):
    r_idx = state_idx // num_cols
    c_idx = state_idx % num_rows
    return (r_idx, c_idx)

def state_to_idx(state,  num_rows, num_cols):
    return (state[0] * num_cols) + state[1]
