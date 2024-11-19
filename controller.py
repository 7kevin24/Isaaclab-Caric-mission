import torch

PI = 3.1415926

def QuaternionToRotationMatrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        quat (torch.Tensor): Quaternion tensor [batch_size, 4] or [4] (w,x,y,z)

    Returns:
        torch.Tensor: Rotation matrices [batch_size, 3, 3] or [3, 3]
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    x2, y2, z2 = x * 2, y * 2, z * 2
    wx, wy, wz = w * x2, w * y2, w * z2
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2

    R = torch.stack([
        torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=1),
        torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=1),
        torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=1)
    ], dim=1)

    return R.squeeze()

def QuaternionToRPY(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to roll, pitch, yaw angles.

    Args:
        quat (torch.Tensor): Quaternion tensor [batch_size, 4] or [4] (w,x,y,z)

    Returns:
        torch.Tensor: Roll, pitch, yaw angles [batch_size, 3] or [3]
    """
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    rpy = torch.stack([roll, pitch, yaw], dim=1)
    return rpy.squeeze()

def RpyToQuaternion(rpy: torch.Tensor) -> torch.Tensor:
    """
    Convert roll, pitch, yaw angles to quaternions.

    Args:
        rpy (torch.Tensor): Roll, pitch, yaw angles [batch_size, 3] or [3]

    Returns:
        torch.Tensor: Quaternion tensor [batch_size, 4] or [4] (w,x,y,z)
    """
    if rpy.dim() == 1:
        rpy = rpy.unsqueeze(0)
        
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    
    quat = torch.stack([w, x, y, z], dim=1)
    return quat.squeeze()

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, device, integral_limit: float = 1.0):
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        self.integral_limit = integral_limit
        self.device = device
        self.integral = None  # Will be initialized on first update
        self.prev_error = None  # Will be initialized on first update
            
    def update(self, error: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Update the PID controller with support for batched inputs.

        Args:
            error (torch.Tensor): Error between desired and current value.
            dt (float): Time step.

        Returns:
            torch.Tensor: Control output.
        """
        # Initialize integral and prev_error with proper shape on first call
        if self.integral is None:
            self.integral = torch.zeros_like(error,device=self.device)
        if self.prev_error is None:
            self.prev_error = torch.zeros_like(error,device=self.device)

        self.integral = self.integral + error * dt
        
        # Apply integral windup limit
        if self.integral_limit is not None:
            self.integral = torch.clamp(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else torch.zeros_like(error)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error.clone()
        
        return output

    def clear_integral(self):
        """Clear the integral term."""
        if self.integral is not None:
            self.integral = torch.zeros_like(self.integral,device=self.device)
        if self.prev_error is not None:
            self.prev_error = torch.zeros_like(self.prev_error,device=self.device)

class Controller:
    """
    Base class for the controllers.

    Args:
        device: Torch device.

    """
    def __init__(self, device):
        self.device = device

    def control(self, *args, **kwargs):
        """
        Compute the control outputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Control outputs.
        """
        raise NotImplementedError

class AttAltController(Controller):
    """
    Attitude and altitude controller for a quadcopter.

    Args:
        robot_mass (float): Mass of the quadcopter.
        J (torch.Tensor): Inertia matrix of the quadcopter.
        device: Torch device.
        gravity (float, optional): Gravitational acceleration. Default is 9.81.
        moment_limit (float, optional): Limit for the moments.
    """
    def __init__(self, robot_mass: float, J: torch.Tensor, device, gravity: float = 9.81, moment_limit: float = 0.007):
        super().__init__(device)
        self.mass = robot_mass
        self.J = J.to(device)  # Ensure J is on correct device
        self.g = gravity
        self.moment_limit = moment_limit

        # PID controllers for roll, pitch, yaw
        self.roll_pid = PIDController(kp=0.5, ki=0.07, kd=0.1, device=device)
        self.pitch_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, device=device)
        self.yaw_pid = PIDController(kp=1.2, ki=0.05, kd=0.1, device=device)
        
        # PID controller for altitude
        self.alt_pid = PIDController(kp=5.0, ki=0.05, kd=1.0, device=device)

    def control(self, 
                desired_attitude: torch.Tensor,
                desired_altitude: torch.Tensor,
                current_attitude: torch.Tensor,
                current_altitude: torch.Tensor, 
                R_robot: torch.Tensor,
                dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the control outputs for attitude and altitude.

        Args:
            desired_attitude (torch.Tensor): Desired roll, pitch, yaw angles.
            desired_altitude (torch.Tensor): Desired altitude.
            current_attitude (torch.Tensor): Current roll, pitch, yaw angles.
            current_altitude (torch.Tensor): Current altitude.
            R_robot (torch.Tensor): Rotation matrix of the robot.
            dt (float): Time step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total thrust and moments.
        """
        batch_size = desired_attitude.shape[0]

        # Calculate attitude errors
        attitude_error = desired_attitude - current_attitude

        # Get PID outputs for attitude
        moments = torch.zeros((batch_size, 3), device=self.device)
        moments[:, 0] = self.roll_pid.update(attitude_error[:, 0], dt)
        moments[:, 1] = self.pitch_pid.update(attitude_error[:, 1], dt)
        moments[:, 2] = self.yaw_pid.update(attitude_error[:, 2], dt)
        
        # Clamp moments using tensor on correct device
        moments = torch.clamp(moments, 
                            torch.tensor(-self.moment_limit, device=self.device),
                            torch.tensor(self.moment_limit, device=self.device))

        # Calculate required thrust considering attitude
        z_world = torch.bmm(R_robot, 
                          torch.tensor([0.0, 0.0, 1.0], device=self.device)
                          .repeat(batch_size, 1)
                          .unsqueeze(2)).squeeze(2)
        
        altitude_error = desired_altitude - current_altitude
        altitude_force = self.alt_pid.update(altitude_error, dt)
        
        # Total thrust needed
        z_dot_product = torch.sum(z_world * torch.tensor([0.0, 0.0, 1.0], device=self.device), dim=1)
        total_thrust = (self.mass * self.g + altitude_force) / (z_dot_product + 1e-6)

        # Create thrust limits on correct device
        thrust_min = torch.tensor(self.mass * self.g * 0.5, device=self.device)
        thrust_max = torch.tensor(self.mass * self.g * 1.5, device=self.device)
        
        # Clamp total thrust
        total_thrust = torch.clamp(total_thrust, thrust_min, thrust_max)

        return total_thrust, moments

class VelocityController(Controller):
    """
    Velocity controller for a quadcopter.

    Args:
        device: Torch device.
        gravity (float, optional): Gravitational acceleration. Default is 9.81.
        velocity_limit (float, optional): Limit for the velocities.
        acceleration_limit (float, optional): Limit for the accelerations.
    """
    def __init__(self, device, gravity: float = 9.81, velocity_limit: float = 5.0, acceleration_limit: float = 2.0):
        super().__init__(device)
        self.ax_pid = PIDController(kp=1.0, ki=0.15, kd=0.1, device=device)
        self.ay_pid = PIDController(kp=1.0, ki=0.15, kd=0.1, device=device)
        self.az_pid = PIDController(kp=1.0, ki=0.15, kd=0.2, device=device)
        self.gravity = gravity
        self.velocity_limit = velocity_limit
        self.acceleration_limit = acceleration_limit
    
    def control(self, desired_velocity: torch.Tensor, current_velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute the desired attitude based on velocity control.

        Args:
            desired_velocity (torch.Tensor): Desired velocities [vx, vy, vz].
            current_velocity (torch.Tensor): Current velocities [vx, vy, vz].
            dt (float): Time step.

        Returns:
            torch.Tensor: Desired roll, pitch, yaw angles.
        """
        batch_size = desired_velocity.shape[0]
        
        # Clamp desired velocities
        desired_velocity = torch.clamp(desired_velocity, -self.velocity_limit, self.velocity_limit)

        # Calculate accelerations for each axis
        acc_dir = torch.zeros((batch_size, 3), device=self.device)
        acc_dir[:, 0] = self.ax_pid.update(desired_velocity[:, 0] - current_velocity[:, 0], dt)
        acc_dir[:, 1] = self.ay_pid.update(desired_velocity[:, 1] - current_velocity[:, 1], dt)
        acc_dir[:, 2] = self.az_pid.update(desired_velocity[:, 2] - current_velocity[:, 2], dt)
        acc_dir[:, 2] += self.gravity

        # Clamp accelerations
        acc_dir = torch.clamp(acc_dir, -self.acceleration_limit, self.acceleration_limit)

        # Normalize acceleration direction
        acc_dir = acc_dir / (torch.norm(acc_dir, dim=1, keepdim=True) + 1e-6)

        # Calculate desired attitude
        desired_attitude = torch.zeros((batch_size, 3), device=self.device)
        desired_attitude[:, 0] = -torch.atan2(acc_dir[:, 1], acc_dir[:, 2])  # Roll
        desired_attitude[:, 1] = torch.atan2(acc_dir[:, 0], 
                                           torch.sqrt(acc_dir[:, 1]**2 + acc_dir[:, 2]**2))  # Pitch
        desired_attitude[:, :2] = torch.clamp(desired_attitude[:, :2], -PI/4, PI/4)  # Limit roll/pitch
        desired_attitude[:, 2] = 0.0  # Yaw

        return desired_attitude

class PositionController(Controller):
    """
    Position controller for a quadcopter.

    Args:
        device: Torch device.
        velocity_limit (float, optional): Limit for the desired velocities.
    """
    def __init__(self, device, velocity_limit: float = 5.0):
        super().__init__(device)
        self.ax_pid = PIDController(kp=5.1, ki=0.07, kd=4.0, device=device, integral_limit=5.0)
        self.ay_pid = PIDController(kp=5.1, ki=0.07, kd=4.0, device=device, integral_limit=5.0)  
        self.velocity_limit = velocity_limit

    def control(self, 
                desired_position: torch.Tensor, 
                current_position: torch.Tensor, 
                rotation: torch.Tensor,
                dt: float) -> torch.Tensor:
        """
        Compute the desired velocity based on position control.

        Args:
            desired_position (torch.Tensor): Desired position [batch, 3]
            current_position (torch.Tensor): Current position [batch, 3]
            rotation (torch.Tensor): Rotation quaternion [batch, 4]
            dt (float): Time step

        Returns:
            torch.Tensor: Desired velocities in body frame [batch, 3]
        """
        batch_size = desired_position.shape[0]
        
        # Calculate position errors - only for x,y
        position_error = desired_position[:, :2] - current_position[:, :2]  # [batch, 2]
        
        # Initialize desired velocity in world frame
        desired_velocity_w = torch.zeros((batch_size, 3), device=self.device)
        
        # Update PIDs for each axis
        desired_velocity_w[:, 0] = self.ax_pid.update(position_error[:, 0], dt)
        desired_velocity_w[:, 1] = self.ay_pid.update(position_error[:, 1], dt)
        
        # Convert quaternion to rotation matrix for each batch element
        R = torch.zeros((batch_size, 3, 3), device=self.device)
        for i in range(batch_size):
            R[i] = QuaternionToRotationMatrix(rotation[i])
        
        # Convert to body frame for each batch element
        desired_velocity_b = torch.bmm(R.transpose(1, 2), desired_velocity_w.unsqueeze(-1)).squeeze(-1)
        
        # Clamp desired velocities
        desired_velocity_b = torch.clamp(desired_velocity_b, -self.velocity_limit, self.velocity_limit)

        return desired_velocity_b