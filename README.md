# orbit_tutorial_felix
I'm trying to grasp Isaac Orbit. Here is just some envs I created to learn Orbit.

## 2024.03.02
Upload orbit_velocity_tracking_env_cfg file. 

TO DO: 
- Check if it works anywhere under the orbit environment; 
- Check the train.py and play.py of the orbit repo; 
- Try to visualize contact force by setting attribute of ContactSensorCfg'set_debug_vis' to True.
- Understand the rsl-rl implementation by reading the source code. By the way, review PPO (and RL).

## 2024.03.08
Upload orbit_velocity_tracking_env_cfg/run_env.py to enable visualization of created RL environment for debugging.

You can see the env.step() log:

-----DICTIONARY OF OUTPUT-----<br>
 ({'policy': tensor([[-0.0157, -0.0024,  0.4098,  ..., -0.2987, -0.2533, -0.1537],
        [-0.1548,  0.4646,  0.0430,  ..., -0.2722, -0.2613, -0.2128],
        [-0.1390, -0.1175, -0.1845,  ..., -0.2223, -0.2633, -0.1676],
        ...,
        [-0.1212,  0.0274,  0.0838,  ..., -0.3071, -0.1395, -0.1751],
        [-0.1393,  0.0231,  0.0561,  ..., -0.2823, -0.2131, -0.2792],
        [ 0.0677,  0.1135,  0.0966,  ..., -0.3460, -0.3559, -0.3257]],
       device='cuda:0')}, tensor([-0.0092, -0.0022, -0.0063,  0.0084, -0.0126,  0.0178,  0.0081,  0.0213,
        -0.0149, -0.0074,  0.0245, -0.0166,  0.0024, -0.0110,  0.0093, -0.0350,
        -0.0205,  0.0137, -0.0121, -0.0089, -0.0126, -0.0210, -0.0041, -0.0102,
         0.0035, -0.0157,  0.0222, -0.0164, -0.0157,  0.0051,  0.0117,  0.0312],
       device='cuda:0'), tensor([False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False], device='cuda:0'), tensor([False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False], device='cuda:0'), {'log': {'Episode Reward/track_lin_vel_xy_exp': tensor(0.0030, device='cuda:0'), 'Episode Reward/track_ang_vel_z_exp': tensor(0.0041, device='cuda:0'), 'Episode Reward/lin_vel_z_l2': tensor(-0.0293, device='cuda:0'), 'Episode Reward/ang_vel_xy_l2': tensor(-0.0051, device='cuda:0'), 'Episode Reward/dof_torques_l2': tensor(-0.0010, device='cuda:0'), 'Episode Reward/dof_acc_l2': tensor(-0.0039, device='cuda:0'), 'Episode Reward/action_rate_l2': tensor(-0.0029, device='cuda:0'), 'Episode Reward/feet_air_time': tensor(-3.4000e-06, device='cuda:0'), 'Episode Reward/flat_orientation_l2': tensor(0., device='cuda:0'), 'Episode Reward/dof_pos_limits': tensor(0., device='cuda:0'), 'Curriculum/terrain_levels': 4.03125, 'Metrics/base_velocity/error_vel_xy': 0.01883886568248272, 'Metrics/base_velocity/error_vel_yaw': 0.016240473836660385, 'Episode Termination/time_out': 0, 'Episode Termination/base_contact': 1}}) <br>
-----END OF OUTPUT-----




