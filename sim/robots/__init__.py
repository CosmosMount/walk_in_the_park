from sim.robots.go1 import Go1


def get_robot_cls(robot_name: str):
    name = robot_name.lower()
    if name == 'go1':
        return Go1
    raise ValueError(f'Unsupported robot_name: {robot_name}')


def make_robot(robot_name: str, **kwargs):
    return get_robot_cls(robot_name)(**kwargs)


__all__ = ['Go1', 'get_robot_cls', 'make_robot']
