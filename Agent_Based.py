import numpy as np
import matplotlib.pyplot as plt
import random


# Assumes that the infector determines the rate of infection. Does not depend on location. Only number of people and year
# This class runs the simulation

class Simulator:
    def __init__(self, t_domain, dt, locations, people, lam, gamma, quarantine):
        self.t = np.linspace(t_domain[0], t_domain[1], int((t_domain[1] - t_domain[0])/dt + 1))
        self.locations = locations
        self.dt = dt
        self.lam = lam
        self.gamma = gamma
        self.people = people
        self.quarantine = quarantine
        self.states = np.zeros((len(self.t), len(self.people)))
        self.S, self.I, self.R, self.Loc_Inf = self.simulate()


    def simulate(self):
        recovery = abs(Gaussian(1/self.gamma, 2).sample(len(self.people)))
        loc_inf = np.zeros((len(self.locations)))
        S, I, R = np.zeros((len(self.t))), np.zeros((len(self.t))), np.zeros((len(self.t)))
        num_infected = 0
        for person_ind in range(0,len(self.people)):
            if self.people[person_ind].current_state == 1:
                num_infected = num_infected + 1
                self.states[0][person_ind] = 1

        I[0] = num_infected
        S[0] = len(self.people)-num_infected
        
        for t_ind in range(0, len(self.t)-1):
            for loc_ind in range(0, len(self.locations)):
                location = self.locations[loc_ind]
                ids = [person.id for person in location.people]
                
                # if the location is Quarantine, no infection could occur, only recovery
                if location.name == "Quarantine":
                    for i in range(0, len(ids)):
                        person_id = ids[i]
                        person_state = self.states[t_ind, person_id]
                        days_sick = np.sum(self.states[0:t_ind, person_id])
                        if person_state == 1:
                            if days_sick >= recovery[person_id]:
                                self.states[t_ind + 1, person_id] = 2
                            else:
                                self.states[t_ind + 1, person_id] = 1
                        self.people[person_id].past_locations.append(location)

                else:
                    potential_infs, potential_inf_ids = self.precompute_infections(location)
                    for i in range(0, len(ids)):
                        potential_infector_id = ids[i]
                        # Update the location just to store
                        self.people[potential_infector_id].past_locations.append(location)
                        self.update_state(t_ind, potential_infector_id, potential_inf_ids[i], recovery)
                    # Update person_1's state itself?
                    
                    
            S[t_ind + 1] = len(np.where(self.states[t_ind + 1 ,:] == 0)[0])
            I[t_ind + 1] = len(np.where(self.states[t_ind + 1, :] == 1)[0])
            R[t_ind + 1] = len(np.where(self.states[t_ind + 1, :] == 2)[0])


            total_pop_size = len(self.people)  # Store total size of people array
            lin_list = np.linspace(0, total_pop_size - 1, total_pop_size)
            # draw_sample = np.random.choice(lin_list, total_pop_size)  # Make linear list random for sampling
            np.random.shuffle(lin_list)
            draw_sample = lin_list
            for loc_ind in range(0, len(self.locations)):
                self.locations[loc_ind].people = []
            # Go through randomized list of people
            for person_id in draw_sample:  # pretend there is no max for each room, invert location and persons so can choose 100# person for a room
                person_id = int(person_id)
                days_sick = np.sum(self.states[0:t_ind, person_id])
                person_state = self.states[t_ind, person_id]
                # if past location == quarantine
                if self.people[person_id].past_locations[-1].name == "Quarantine":
                    # if the person already recovered => he can come out of quarantine
                    if self.states[t_ind + 1, person_id] == 2:
                        new_loc = np.random.choice(self.locations[:-1], 1, p=self.people[person_id].P_transition)[0]
                        if len(new_loc.people) > new_loc.protocol.max_people:
                            raise TypeError("Maximum Capacity exceeded")
                        self.locations[new_loc.ind].people.append(self.people[person_id])
                    # else, he has to stay in quarantine
                    elif self.states[t_ind + 1, person_id] == 1:
                        self.locations[self.people[person_id].past_locations[-1].ind].people.append(self.people[person_id])
                    else:
                        raise TypeError("An uninfected person is in quarantine")
                # the past location is not quarantine
                else:
                    # if the person met these conditions, we sent him in quarantine
                    if self.quarantine and person_state == 1 and days_sick >= AVG_DAYS_SICK and self.states[t_ind + 1, person_id] != 2 and random.uniform(0, 1)<=0.8 and self.people[person_id].past_locations[-1].name != "Quarantine":
                        new_loc = self.locations[-1]
                    # else, we randomly assign people to other locations
                    else:
                        new_loc = np.random.choice(self.locations[:-1], 1, p=self.people[person_id].P_transition)[0]
                    if len(new_loc.people) > new_loc.protocol.max_people:
                        raise TypeError("Maximum Capacity exceeded")
                    self.locations[new_loc.ind].people.append(self.people[person_id])
                    
                

        return S, I, R, loc_inf

    def precompute_infections(self, location):
        ids = [person.id for person in location.people]
        infections = location.protocol.P.sample(len(ids))
        inf_mat = []
        # Assign who you will infect
        for i in range(0, len(ids)):
            infector_id = ids[i]
            inf_ids = random.choices(ids, k=infections[i])
            inf_mat.append(list(inf_ids))
            # Might pick the person themselves, or the same person twice. But this shouldn't be a problem
        return infections, inf_mat

    def update_state(self, t_ind, person_1_id, person_2_ids, recovery):
        #Update the state by checking for the collision

        person_1_state = self.states[t_ind, person_1_id]
        if person_1_state == 0:
            return
        if person_1_state == 2:
            self.states[t_ind+1][person_1_id] = 2
            return

        for j in range(0, len(person_2_ids)):
            person_2_id = person_2_ids[j]
            person_2_state = self.states[t_ind, person_2_id]

            if person_1_id == person_2_id:
                continue
            if person_2_state == 1:
                continue
            if person_2_state == 2:
                self.states[t_ind + 1][person_2_id] = 2
                continue
            if self.states[t_ind+1, person_2_id] == 1:
                continue
            #None of the other if statements were triggered, so this is a successful infection
            self.states[t_ind + 1,person_2_id] = 1
        #Check recovery

        days_sick = np.sum(self.states[0:t_ind, person_1_id])
        if person_1_state == 1:
            if days_sick >= recovery[person_1_id]:
                self.states[t_ind + 1, person_1_id] = 2
            else:
                self.states[t_ind + 1, person_1_id] = 1



# This class represents each person
class Person:
    def __init__(self, year, current_state, P_transition, id):
        self.year = year
        self.current_state = current_state #int of 0 (susceptible), 1 (infected), or 2 (recovered)
        self.past_locations = [] # Stores all the prior locations visited at each time step
        self.P_transition = P_transition # vector of length locations
        self.id = id


# This class represents each location (an associated ID and name)
class Location:
    def __init__(self, ind, name, protocol, people):
        self.ind = ind #int of (0, 100) for their location
        self.name = name #string that represents the name
        self.protocol = protocol #Restrictions/policies put in place
        self.people = people

# This class represents protocols for each location (i.e. max people, spacing, masks. etc.)
# Still need to implement a function that takes into account these protocols and defined probability parameters
    # For example: Masks might decrease the mean of the gaussian that models the infection probability
class Protocol:
    def __init__(self, max_people, P_type, lam_scale_fac):
        self.max_people = max_people
        self.P_type = P_type
        self.lam_scale_fac = lam_scale_fac
        parameters = [lam*lam_scale_fac]
        self.P = self.define_probability(parameters)

    def define_probability(self, params):
        if self.P_type == "Gaussian":
            return Gaussian(params[0], params[1])

        if self.P_type == "Poisson":
            return Poisson(params[0])

        if self.P_type == "Beta":
            return Beta(params[0], params[1])

        if self.P_type == "Lognormal":
            return Lognormal(params[0], params[1])

        raise Exception("P_type was not in selected list")


#The below class is a SIR Explicit
class SIR:
    def __init__(self, beta, gamma, N, t_domain, dt, S0, I0, R0):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.t_domain = t_domain
        self.dt = dt
        self.t =t_vec = np.linspace(t_domain[0], t_domain[1], int((t_domain[1] - t_domain[0])/dt + 1))
        self.S = np.zeros(len(self.t))
        self.I = np.zeros(len(self.t))
        self.R = np.zeros(len(self.t))
        self.S[0] = S0
        self.I[0] = I0
        self.R[0] = R0
    def explicitSolve(self):

        for t_ind in range(0, len(self.t)-1):
            self.S[t_ind+1] = self.S[t_ind] -  self.beta*self.S[t_ind]*self.I[t_ind]/self.N * self.dt
            self.I[t_ind+1] = self.I[t_ind] + (self.beta*self.S[t_ind]*self.I[t_ind]/self.N - self.gamma*self.I[t_ind])*self.dt
            self.R[t_ind+1] = self.R[t_ind] +  self.gamma*self.I[t_ind]*self.dt



# The below classes are probability distributions
class Gaussian:
    def __init__(self, mu, st_dev):
        self.mu = mu
        self.st_dev = st_dev
        self.type = "Gaussian"

    def sample(self, N):
        samples = np.random.normal(self.mu, self.st_dev, N)
        return samples

    def plot(self, num_samples, bins, title, num_rows, num_cols, ind):
        samples = self.sample(num_samples)
        plt.subplot(num_rows, num_cols, ind)
        plt.hist(samples, bins, density=True)
        plt.title(title)

class Poisson:
    def __init__(self, lam):
        self.lam = lam
        self.type = "Poisson"

    def sample(self, N):
        samples = np.random.poisson(self.lam, N)
        return samples

    def plot(self, num_samples, bins, title, num_rows, num_cols, ind):
        samples = self.sample(num_samples)
        plt.subplot(num_rows, num_cols, ind)
        plt.hist(samples, bins, density=False)
        plt.title(title)

class Beta:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self, N):
        samples = np.random.beta(self.a, self.b, N)
        return samples

    def plot(self, num_samples, bins, title, num_rows, num_cols, ind):
        samples = self.sample(num_samples)
        plt.subplot(num_rows, num_cols, ind)
        plt.hist(samples, bins, density=True)
        plt.title(title)

class Lognormal:
    def __init__(self, mu, st_dev):
        self.mu = mu
        self.st_dev = st_dev

    def sample(self, N):
        samples = np.random.lognormal(self.mu, self.st_dev, N)
        return samples

    def plot(self, num_samples, bins, title, num_rows, num_cols, ind):
        samples = self.sample(num_samples)
        plt.subplot(num_rows, num_cols, ind)
        plt.hist(samples, bins, density=True)
        plt.title(title)

beta = 5/6.5
lam = 3.2/6.5
gamma = 1/6.5
dt = 1
AVG_DAYS_SICK = 3
total_people = 3346  # See spreadsheet
expected_infected = 228  # See spreadsheet



# This is just to test the above classes
people = []
for p in range(0, total_people):
    people.append(Person(year="Undergrad", current_state=0, P_transition=[0.25, 0.25, 0.25, 0.25], id=p))

for x in range(0, expected_infected):
    people[x].current_state = 1

type = "Poisson"
loc_1 = Location(ind=0, name="Residential", protocol=Protocol(max_people=10e5, lam_scale_fac=1, P_type=type), people=people[0:total_people])
loc_2 = Location(ind=1, name="Gym", protocol=Protocol(max_people=10e5, lam_scale_fac=1, P_type=type), people=[])
loc_3 = Location(ind=2, name="Dining Hall", protocol=Protocol(max_people=10e5, lam_scale_fac=1, P_type=type), people=[])
loc_4 = Location(ind=3, name="Oval", protocol=Protocol(max_people=10e5, lam_scale_fac=1, P_type=type), people=[])
Quarantine = Location(ind=4, name="Quarantine", protocol=Protocol(max_people=10e5, lam_scale_fac=1, P_type=type), people=[])
locations = [loc_1 , loc_2, loc_3, loc_4, Quarantine] ##### MUST ADD LOCATIONS IN ORDER OF IND


sim = Simulator(t_domain=[0, 100], dt=dt, locations=locations, people=people, lam=lam, gamma=gamma, quarantine=True)
ind_max_I = np.argmax(sim.I)
max_I = np.max(sim.I)
beta_estimate = gamma * total_people/sim.S[ind_max_I]
print(sim.S[ind_max_I])
print(beta_estimate)


# Binary Search
b0 = beta
b0min = 0
b0max= 3*beta
for i in range(0, 10):
    SIR_Continuous = SIR(beta=b0, gamma=gamma, N=len(people), t_domain=[0, 100], dt=0.1, S0=total_people-expected_infected, I0=expected_infected, R0=0)
    SIR_Continuous.explicitSolve()
    max_I_test = np.max(SIR_Continuous.I)
    if max_I_test > max_I:
        b0max = b0
    if max_I_test < max_I:
        b0min = b0
    b0 = (b0min + b0max)/2
    print(b0)





plt.figure()
plt.plot(sim.t, sim.S, 'y')
plt.plot(sim.t, sim.I, 'r')
plt.plot(sim.t, sim.R, 'b')
plt.plot(SIR_Continuous.t, SIR_Continuous.S, '--y')
plt.plot(SIR_Continuous.t, SIR_Continuous.I, '--r')
plt.plot(SIR_Continuous.t, SIR_Continuous.R, '--b')
plt.legend(("S_Agent", "I_Agent", "R_Agent", "S_Cont", "I_Cont", "R_Cont"))
plt.suptitle("Stanford Agent-Based Plot vs Continuous")
plt.xlabel("Time (days)")
plt.ylabel("People")
plt.title("Beta Equivalent = " + str("{:.2f}".format(round(b0, 2))) + " Lam = " + str("{:.2f}".format(round(lam, 2)))+ ", Gamma = " + str("{:.2f}".format(round(gamma, 2))) + ", Days Sick = " + str(AVG_DAYS_SICK))
plt.show()

g = Gaussian(0, 0.1)
p = Poisson(0.5)
b = Beta(8, 2)
l = Lognormal(0, 0.1)
g.plot(num_samples=10000, bins=100, title='Gaussian', num_rows=2, num_cols=2, ind=1);
p.plot(num_samples=10000, bins=100, title='Poisson', num_rows=2, num_cols=2, ind=2);
b.plot(num_samples=10000, bins=100, title='Beta', num_rows=2, num_cols=2, ind=3);
l.plot(num_samples=10000, bins=100, title='Lognormal', num_rows=2, num_cols=2, ind=4);
#plt.show()