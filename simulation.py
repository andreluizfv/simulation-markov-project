import numpy as np
from statistics import mean
# Define the state space
states = ["late", "ok",
          "heavy_jam", "light_jam", "no_jam_late", "no_jam_ok",
          "no_acc_jam", "acc_jam", "no_acc_no_jam_late", "no_acc_no_jam_ok",
          "jam_open", "jam_closed", "no_jam_late_open", "no_jam_late_closed", "no_jam_ok_open", "no_jam_ok_closed"]

# Define the transition matrix
def get_transition_matrix(flux, rain):
    p_light_jam = flux['p_light']
    p_heavy_jam = flux['p_heavy']
    p_no_jam = flux['p_no']
    p_light_jam_to_acc = p_acc * 3.7 # font: news
    p_heavy_jam_to_acc = p_acc * 6.3
    p_closed = 15/125
    p_open = 1 - p_closed
    if rain:
        p_light_jam_to_acc *= 2.07
        p_heavy_jam_to_acc *= 2.07
    return np.array([[0, 0, p_heavy_jam, p_light_jam, p_no_jam, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                     [0, 0, p_heavy_jam, p_light_jam, 0, p_no_jam, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, (1 - p_heavy_jam_to_acc), p_heavy_jam_to_acc, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, (1 - p_light_jam_to_acc), p_light_jam_to_acc, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_open, p_closed, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_open, p_closed, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_open, p_closed, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_open, p_closed],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #student thinks they will arrive on time since everything is going well
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],]) 

yellow_flux = { # font: pool with people that uses car to go to school everyday
    'p_light': 0.7,
    'p_heavy': 0.1,
    'p_no': 0.2,
}
red_flux = {
    'p_light': 0.15,
    'p_heavy': 0.8,
    'p_no': 0.05,
}
blue_flux = {
    'p_light': 0.14,
    'p_heavy': 0.01,
    'p_no': 0.85,
}

p_acc = mean([0.05, 0.07, 0.005, 0.005, 0.0075, 0.01]) # chance of accident
p_start_late = 0.3 # 30% starting late, 70% starting on time
fluxes = [blue_flux, yellow_flux, red_flux]
def run_sample(fluxes, rain):
    # Initial state
    current_state = np.random.choice(states, p=[p_start_late, 1-p_start_late, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
    # Simulate transitions
    num_steps = 4
    simulated_states = [current_state]
    for i in range(num_steps*len(fluxes)):
        current_state = np.random.choice(states, p=get_transition_matrix(fluxes[int(i/4)], rain)[states.index(current_state)])
        simulated_states.append(current_state)
    print("Simulated States:", simulated_states)
    return simulated_states

hurry_factor = 0.8 #how fast student go if he thinks he is late
rain_factor = 0.9 #downgrade in speed due rain (internet fonts said it should be 80% of the speed, but in the target route there is no space to many free road where it should apply hard)
def get_final_time(avr_times, simulated_states, rain):
    if len(avr_times) != ((len(simulated_states) - 1)/4):
        return "error: number of time intervals incompatible with these simulated_states"
    final_time = 0
    if simulated_states[0] == "late":
        final_time += 5
        avr_times[0] *= hurry_factor
    for happening in simulated_states[1:]:
        match happening:
            case "heavy_jam":
                avr_times[0] *= 1.1
            case "no_jam_late" | "no_jam_ok":
                avr_times[0] *= 0.9
            case "acc_jam":
                avr_times[0] *= 1.32
            case "jam_closed" | "no_jam_late_closed" | "no_jam_ok_closed":
                avr_times[0] += 0.125 # I took a sample of 15s delay in semaphore, where
            case "ok":
                final_time += avr_times[0]
                avr_times.pop(0)
            case "late":
                final_time += avr_times[0]
                avr_times.pop(0)
                if len(avr_times) > 0: avr_times[0]*hurry_factor
            case _:
                continue
    return final_time if not rain else final_time/0.9
rain = True
simulated_states = run_sample(fluxes, rain)
print(get_final_time([4, 5, 6], simulated_states, rain))