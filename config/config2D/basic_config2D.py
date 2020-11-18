m_dim = 2
# resolution for the simulation grid
m_res = [600, 600]
# resolution for the screen space
m_screen_res = [600, 600]

# a pixel length, we assume the pixel is always square
m_dx = 1.0
m_inv_dx = 1.0 / m_dx
m_half_inv_dx = 0.5 * m_inv_dx
m_dt = 0.03
m_half_dt = m_dt / 2