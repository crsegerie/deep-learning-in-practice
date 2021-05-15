import argparse
# from script.interpolate import interp1d
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = (10000//self.delta_t)
        # Loading the model
        PATH = "final_model"
        print("Loading :", PATH)
        from ode_model import ODEFunc
        model = ODEFunc()
        model.load_state_dict(torch.load(PATH))
        model.eval()
        self.rosler_nn = model

        self.initial_condition = np.array(value.init)

    def full_traj(self,initial_condition=None):
        if initial_condition is None:
            initial_condition = np.array(value.init)

        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        from scipy.integrate import solve_ivp

        def v_eq_pred(t, y0):
            with torch.no_grad():
                y0 = torch.tensor(y0.astype(np.float32))
                y0 = torch.reshape(y0, (1, 3))
                return self.rosler_nn(y0)[0].numpy()

        def pred_traj():
            t = np.linspace(0., 10000, int(10000/self.delta_t))
            return solve_ivp(v_eq_pred, [0, len(t)], initial_condition, method='RK45', t_eval=t)["y"]

        pred_y = pred_traj()

        # Extracting the y component
        y = pred_y[1, :]
        print("Finished the prediction of the trajectory with shape:", y.shape)

        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy',y)
        
    
if __name__ == '__main__':
    delta_t = 1e-2
    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)

