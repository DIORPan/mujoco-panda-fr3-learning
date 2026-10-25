import mujoco


def build_fr3_model(with_scene=False):
    """组合 FR3 与 Panda 夹爪，并添加名义抓取点。"""
    fr3 = mujoco.MjSpec.from_file("assets/franka_fr3/fr3.xml")
    hand = mujoco.MjSpec.from_file("assets/franka_emika_panda/hand.xml")

    hand.body("hand").quat = [0.9238795, 0, 0, -0.3826834]

    hand.body("hand").add_site(
        name="grasp_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0, 0, 0.1034],
        size=[0.008, 0, 0],
        rgba=[1, 0, 1, 1],
        group=0,
    )

    fr3.attach(hand, prefix="panda_", site="attachment_site")
    if with_scene:
        environment = mujoco.MjSpec.from_file(
            "assets/franka_fr3/task_environment.xml"
        )
        fr3.attach(
            environment,
            prefix="",
            frame=fr3.worldbody.add_frame(),
        )
        fr3.stat.center[:] = environment.stat.center
        fr3.stat.extent = environment.stat.extent
    return fr3.compile()