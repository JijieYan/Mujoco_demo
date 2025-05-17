## Installation

```
pip install mujoco
pip install numpy
```

## 运行

```
python base_env.py
```

## In Mujoco，Franka Emika 控制说明

### 夹爪控制

机器人的夹爪由两个手指组成，可以打开或关闭以抓取物体。这一功能通过`gripper()`方法实现：

```python
def gripper(self, open=True):
    self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
    self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]
```

#### 参数：

- `open` (布尔值)：控制夹爪状态
  - `True`（默认）：打开夹爪，将两个手指关节位置设置为0.04
  - `False`：关闭夹爪，将两个手指关节位置设置为0

#### 使用

```python
    self.gripper(False)
```
给gripper函数传入`True`或者`False`参数来控制夹爪状态

### 机械臂控制

7dof机械臂通过在`control()`方法中实现的PD控制器进行控制,该方法计算将末端执行器移动到期望位置和方向所需的力矩：

```python
def control(self, xpos_d, xquat_d):
    xpos = self.data.body("panda_hand").xpos
    xquat = self.data.body("panda_hand").xquat
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
    mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)
    error = np.zeros(6)
    error[:3] = xpos_d - xpos
    res = np.zeros(3)
    mujoco.mju_subQuat(res, xquat, xquat_d)
    mujoco.mju_rotVecQuat(res, res, xquat)
    error[3:] = -res
    J = np.concatenate((jacp, jacr))
    v = J @ self.data.qvel
    for i in range(1, 8):
        dofadr = self.model.joint(f"panda_joint{i}").dofadr
        self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
            f"panda_joint{i}"
        ).qfrc_bias
        self.data.actuator(f"panda_joint{i}").ctrl += (
            J[:, dofadr].T @ np.diag(self.K) @ error
        )
        self.data.actuator(f"panda_joint{i}").ctrl -= (
            J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
        )
```

### 传入参数：

- `xpos_d` (numpy.ndarray)：末端执行器的期望3D位置[x, y, z]
- `xquat_d` (numpy.ndarray)：末端执行器的期望方向，以四元数[w, x, y, z]表示

#### 使用

```python
    self.control(self.xpos_d, self.xquat0)
    
    # 物理步进
    mujoco.mj_step(self.model, self.data)
```
将期望的pos传入control函数，该函数会遍历并修改每一个joint的位置，随后通过mujoco的mj_step函数进行步进以实现位姿更新。
