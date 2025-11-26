class LAUVControlAllocator:
    def __init__(self):
        # TODO: parameter setup
        # TODO: actuator setup
        # TODO: Casadi setup
        # TODO: subscription
        pass

    def calculate_B_matrix(self):
        # TODO: define actuator model
        # TODO: define fin force
        # TODO: define cost function
        # TODO: create solver
        pass

    def control_callback(self):
        # TODO: convert control command
        # TODO: store velocity command
        pass

    def allocate(self):
        # this will run periodically
        # TODO: setup solver inputs
        # TODO: solve
        # TODO: extract output
        pass

    def publish_control_cmd(self):
        # TODO: publish thruster
        # TODO: publisj fins
        pass