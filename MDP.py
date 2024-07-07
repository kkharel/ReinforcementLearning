

import numpy as np
np.set_printoptions(threshold=np.inf)


# Initialize q values with all zeros for 3x3 matrix which is the initial probability distribution 
size = m = 3
initial_probability = np.zeros(size**2)
initial_probability[size**2//2] = 1 # put the robot at location 4 that is at position (1,1) center.
print("The size of the grid is:", m, "x" , m)
initial_probability_matrix = initial_probability.reshape(m, m)
# print("The initial probability distribution of environment is, 'q':\n", initial_probability_matrix)
initial_probability_matrix

steps = 100000


def get_transition_probabilities(m, p_up, p_down, p_left, p_right):
  if m <= 1 or not np.isclose(p_up + p_down + p_left + p_right, 1.0):
    raise ValueError("Invalid input")

  transition_probability = np.zeros((m**2, m**2))

  states = {}
  for state in range(m**2):  
    states[state+1] = (state // m, state % m)

  for initial_state in range(m**2): 
    for destination_state in range(m**2): 
      row_initial_state, column_initial_state = states[initial_state+1]
      row_destination_state, column_destination_state = states[destination_state+1]

      row_difference = row_initial_state - row_destination_state 
      column_difference = column_initial_state - column_destination_state
      
      #print(f"Initial state: ({r_i}, {c_i}), Destination state: ({r_j}, {c_j}), Row Difference: {row_difference}, Column Difference: {column_difference}")
      
      #horizontal movement
      if row_difference == 0: # no movement row wise
        if column_difference == 1: # column movement to left 
          transition_probability[initial_state, destination_state] = p_left 
        elif column_difference == -1:  # column movement to right
          transition_probability[initial_state, destination_state] = p_right
          
        # check boundaries up,down,left,right to ensure the agent does not go out of bounds
        elif column_difference == 0: # no movement column wise
          if row_initial_state == 0: # top row
            transition_probability[initial_state, destination_state] += p_up # increment with p_up
          elif row_initial_state == m - 1: # bottom row
            transition_probability[initial_state, destination_state] += p_down # increment p_down
          if column_initial_state == 0: # leftmost column
            transition_probability[initial_state, destination_state] += p_left # increment p_left
          elif column_initial_state == m - 1: # rightmost column
            transition_probability[initial_state, destination_state] += p_right #  increment p_rgt
     
      # Vertical Movement 
      elif row_difference == 1: # movement up row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_up 
      elif row_difference == -1: # movement down row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_down
          
  return transition_probability



transition_probability = get_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
timestep = 0
transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, timestep)
print("Transition Probabilities that governs the system at time step 0:\n", np.round(transition_probability_at_timestep_t,3))
print("State probabilities at time step 0:", np.matmul(initial_probability, transition_probability_at_timestep_t))


transition_probability = get_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
timestep = 1
transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, timestep)
print("Transition Probabilities that governs the system at time step 1:\n", np.round(transition_probability_at_timestep_t,3))
print("State probabilities at time step 1:", np.matmul(initial_probability, transition_probability_at_timestep_t))


transition_probability = get_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
timestep = 2
transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, timestep)
print("Transition Probabilities that governs the system at time step 2:\n", np.round(transition_probability_at_timestep_t,3))
print("State probabilities at time step 2:", np.matmul(initial_probability, transition_probability_at_timestep_t))


transition_probability = get_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
timestep = 3
transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, timestep)
print("Transition Probabilities that governs the system at time step 3:\n", np.round(transition_probability_at_timestep_t,3))
print("State probabilities at time step 3:", np.matmul(initial_probability, transition_probability_at_timestep_t))




transition_probability = get_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
time_steps = [0, 1, 3, 5, 10, 20, 100]

# Calculate and print transition probabilities for each time step
for time in time_steps:
  transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, time)
  #print(f"Transition Probabilities that govern the system at time step {t}:\n", np.round(Pt, 3))
  print(f"State probabilities at time step {time}:", np.round(np.matmul(initial_probability, transition_probability_at_timestep_t),2))
  print("\n")


transition_probability = get_transition_probabilities(m=3, p_up=0.25, p_down=0.25, p_left=0.25, p_right=0.25) 
time_steps = [0, 1, 3, 5, 10, 20, 100]

# Calculate and print transition probabilities for each time step
for time in time_steps:
  transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, time)
  #print(f"Transition Probabilities that govern the system at time step {t}:\n", np.round(Pt, 3))
  print(f"State probabilities at time step {time}:", np.round(np.matmul(initial_probability, transition_probability_at_timestep_t),2))
  print("\n")



# Modifying above transition probability function to include absorbing state

def modified_transition_probabilities(m, p_up, p_down, p_left, p_right):
  if m <= 1 or not np.isclose(p_up + p_down + p_left + p_right, 1.0):
    raise ValueError("Invalid input")

  transition_probability = np.zeros((m**2+1, m**2+1)) # addition of crashed state

  states = {}
  for state in range(m**2):  
    states[state+1] = (state // m, state % m)

  for initial_state in range(m**2): 
    for destination_state in range(m**2): 
      row_initial_state, column_initial_state = states[initial_state+1]
      row_destination_state, column_destination_state = states[destination_state+1]

      row_difference = row_initial_state - row_destination_state 
      column_difference = column_initial_state - column_destination_state
      
      #print(f"Initial state: ({r_i}, {c_i}), Destination state: ({r_j}, {c_j}), Row Difference: {row_difference}, Column Difference: {column_difference}")
      
      #horizontal movement
      if row_difference == 0: # no movement row wise
        if column_difference == 1: # column movement to left 
          transition_probability[initial_state, destination_state] = p_left 
        elif column_difference == -1:  # column movement to right
          transition_probability[initial_state, destination_state] = p_right
          
        # check boundaries up,down,left,right to ensure the agent does not go out of bounds
        elif column_difference == 0: # no movement column wise
          if row_initial_state == 0: # top row
            transition_probability[initial_state, destination_state] += p_up # increment with p_up
          elif row_initial_state == m - 1: # bottom row
            transition_probability[initial_state, destination_state] += p_down # increment p_down
          if column_initial_state == 0: # leftmost column
            transition_probability[initial_state, destination_state] += p_left # increment p_left
          elif column_initial_state == m - 1: # rightmost column
            transition_probability[initial_state, destination_state] += p_right #  increment p_rgt
     
      # Vertical Movement 
      elif row_difference == 1: # movement up row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_up 
      elif row_difference == -1: # movement down row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_down
          
  # set the element in the last column of each row to be equal to the original diagonal element.
  # Then set the original diagonal element to zero since the robot will crash now instead of staying on same state
  for i in range(m**2):
    transition_probability[i, m**2] = transition_probability[i, i]
    transition_probability[i,i] = 0
  
  transition_probability[m**2, m**2] = 1 # crashed/absorbing state transitions to itself 

          
  return transition_probability


new_state = 0
initial_probability = np.append(initial_probability, new_state)
initial_probability

# Changing the time steps in gradually increasing order, we see that the robot 
# crash probability is highest. At time step 100, we are 100% certain to see 
# robot in crashed state


transition_probability = modified_transition_probabilities(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35) 
timestep = 100
transition_probability_at_timestep_t = np.linalg.matrix_power(transition_probability, timestep)
print("Transition Probabilities that governs the system at time step 0:\n", np.round(transition_probability_at_timestep_t,3))
print("State probabilities at time step 0:", np.matmul(initial_probability, transition_probability_at_timestep_t))


### Getting/collecting rewards for robot until it crashes 

def modified_transition_probabilities(m, p_up, p_down, p_left, p_right, timestep):
  if m <= 1 or not np.isclose(p_up + p_down + p_left + p_right, 1.0):
    raise ValueError("Invalid input")

  transition_probability = np.zeros((m**2+1, m**2+1)) # addition of crashed state

  states = {}
  for state in range(m**2):  
    states[state+1] = (state // m, state % m)

  for initial_state in range(m**2): 
    for destination_state in range(m**2): 
      row_initial_state, column_initial_state = states[initial_state+1]
      row_destination_state, column_destination_state = states[destination_state+1]

      row_difference = row_initial_state - row_destination_state 
      column_difference = column_initial_state - column_destination_state
      
      #print(f"Initial state: ({r_i}, {c_i}), Destination state: ({r_j}, {c_j}), Row Difference: {row_difference}, Column Difference: {column_difference}")
      
      #horizontal movement
      if row_difference == 0: # no movement row wise
        if column_difference == 1: # column movement to left 
          transition_probability[initial_state, destination_state] = p_left 
        elif column_difference == -1:  # column movement to right
          transition_probability[initial_state, destination_state] = p_right
          
        # check boundaries up,down,left,right to ensure the agent does not go out of bounds
        elif column_difference == 0: # no movement column wise
          if row_initial_state == 0: # top row
            transition_probability[initial_state, destination_state] += p_up # increment with p_up
          elif row_initial_state == m - 1: # bottom row
            transition_probability[initial_state, destination_state] += p_down # increment p_down
          if column_initial_state == 0: # leftmost column
            transition_probability[initial_state, destination_state] += p_left # increment p_left
          elif column_initial_state == m - 1: # rightmost column
            transition_probability[initial_state, destination_state] += p_right #  increment p_right
     
      # Vertical Movement 
      elif row_difference == 1: # movement up row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_up 
      elif row_difference == -1: # movement down row wise
        if column_difference == 0: # no column movement
          transition_probability[initial_state, destination_state] = p_down
          
  # set the element in the last column of each row to be equal to the original diagonal element.
  # Then set the original diagonal element to zero since the robot will crash now instead of transitioning to itself
  for i in range(m**2):
    transition_probability[i, m**2] = transition_probability[i, i]
    transition_probability[i,i] = 0
  
  transition_probability[m**2, m**2] = 1 # crashed/absorbing state transitions to itself 

  expected_rewards = np.zeros(m**2)
  
  for state in range(m**2):
    for i in range(timestep):
      crashed = False
      next_state = state
      episode_reward = 0
      #current_timestep = 0
      while not crashed:
        next_state = np.random.choice(m**2+1, p = transition_probability[next_state, :])
        #current_timestep += 1
        if next_state < m**2:
          episode_reward = episode_reward + 1
        else:
          crashed = True
          #print(f"Robot crashed in episode {i + 1} at timestep {current_timestep}")

      expected_rewards[state] = expected_rewards[state] + episode_reward
      
  expected_rewards = expected_rewards / timestep
          
  return transition_probability, expected_rewards



def print_transition_results(m, p_up, p_down, p_left, p_right, timestep):
  transition_probability, expected_rewards = modified_transition_probabilities(m=m, p_up=p_up, p_down=p_down, p_left=p_left, p_right=p_right, timestep=timestep)
  
  print("Initial Transition Probabilities:\n", np.round(transition_probability, 3))
  print("Expected Rewards:\n", expected_rewards)

  timestep_t_transition_probability = np.linalg.matrix_power(transition_probability, timestep)
  print(f"Transition Probabilities that govern the system at time step {timestep}:\n", np.round(timestep_t_transition_probability, 3))


print_transition_results(m=3, p_up=0.3, p_down=0.2, p_left=0.15, p_right=0.35, timestep=100)

