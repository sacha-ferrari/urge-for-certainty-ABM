#This code feature two different kind of computation: the multivariate doctrinal paradox (DP) and a evidence based policy group deliberation (EBP)
#For the sake of readibility, we mention before each block in which of the two model this code block is used. If no comment: in the both of them.
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count


#DP
MODES = ['purely_random', 'purely_random_excluding_irrationals', 'rationaly_random']

#DP
def LOGICAL_AND(x, y):
    return x if x < y else y

#DP
def IS_LOGICS_VIOLATED(x, y, z):
    return (LOGICAL_AND(x, y) - 0.5) * (z - 0.5) < 0

class MultivariateAgent:
    def __init__(self, index, initial_opinion, epsilon, confidence_type, radicalized=False, doctrinal_paradox=False, profession="", self_radicalisation_coefficient=0.1, tau=1):
        self.index = index
        self.opinions = [[op] for op in initial_opinion]
        self.epsilon = epsilon
        self.radicalized = radicalized
        #DP: if an agent has experience irrationality at least once in his life
        self.has_experienced_irrationality = IS_LOGICS_VIOLATED(initial_opinion[0], initial_opinion[1],
                                                                initial_opinion[2]) if doctrinal_paradox else False
        #DP: type of confidence (see global variable MODES)
        self.confidence_type=confidence_type
        #If we want to implement DP or not
        self.doctrinal_paradox=doctrinal_paradox
        #EBP: profession of the agent lead to different behavior (scientist, policy maker...)
        self.profession=profession
        #EBP: global factor for self-radicalisation
        self.self_radicalisation_alpha=self_radicalisation_coefficient
        #EBP: the "deadline" of the agent (urge for certainty)
        self.self_radicalisation_tau=tau

    def next_opinion(self, community, current_time):
        agents=community.agents

        #DP: in the sperical case, we average the opinions if the agents is inside a n-sphere of confidence
        if self.confidence_type=="spherical":
            opinion_sum = [0]*len(self.opinions)
            number_of_agents = 0
            #compute the distance on all the agents by looping on them
            for agent in agents:
                distance = 0
                #then loop on all the opinions of one agent
                for index, opinion in enumerate(self.opinions):
                    distance += math.pow(agent.opinions[index][current_time] - opinion[current_time], 2)
                #chack the Pythagorian distance and asses if an agent is within the sphere
                if math.sqrt(distance) <= self.epsilon:
                    #if the agent is in the sphere, add his opinion to the mean
                    for index, opinion in enumerate(self.opinions):
                        opinion_sum[index] += agent.opinions[index][current_time]
                    number_of_agents += 1
                distance = 0
            # add the new opinion to the agent's opinion
            for index, opinion in enumerate(self.opinions):
                self.opinions[index].append(opinion_sum[index] / number_of_agents)

        #the trivial linear case (like HK's paper)
        if self.confidence_type=="linear":
            #update of all opinions of the agent
            for index, opinion in enumerate(self.opinions):
                opinion_sum = 0
                number_of_agents = 0
                current_opinion=opinion[current_time]
                for agent in agents:
                    #include an other agent's opinion if he is within the bound of confidence
                    if abs(agent.opinions[index][current_time] - current_opinion) <= self.epsilon :
                        if not (agent.profession=="policy-maker" and self.profession =="scientist") :
                            opinion_sum += agent.opinions[index][current_time]
                            number_of_agents += 1
                #EBP: case of a non bayesian piece of evidence: the evidence is counted as an extra agent
                for evidence in community.evidences:
                    if not evidence.bayesian and current_time>evidence.appearance_time and abs(evidence.value[0] - current_opinion) <= self.epsilon :
                        opinion_sum += evidence.value[0]
                        number_of_agents += 1
                new_position=(opinion_sum / number_of_agents)

                #EBP: case of a bayesian piece of evidence: updating with the  bayesian formula
                for evidence in community.evidences:
                    if evidence.bayesian and current_time == evidence.appearance_time:
                        #override previous new position
                        new_position = evidence.likelihoods[0]* current_opinion /(evidence.likelihoods[0]*current_opinion + evidence.likelihoods[1]*(1-current_opinion))
                #EBP: self radicalisation of the opinion after upadating with neighbours and piece of evidence
                shift = 2*(1-new_position)*(new_position-0.5) if new_position>0.5 else 2*(0-new_position)*(0.5-new_position)
                self.opinions[index].append(new_position+shift*self.self_radicalisation_alpha  *(1-math.exp(-current_time/self.self_radicalisation_tau))  )

        #DP: check if the egent is in an irrational state
        if self.doctrinal_paradox and IS_LOGICS_VIOLATED(self.opinions[0][-1], self.opinions[1][-1], self.opinions[2][-1]):
            self.has_experienced_irrationality = True

class Evidence:
    def __init__(self,values,appearance_time, likelihoods , bayesian=True):
        #in the case of non bayesian piece of evidence, an evidence is like angent with an multivariate opinions values. Hence, values is an array.
        self.value = values
        self.appearance_time = appearance_time
        #likelihoods is an list of two real numbers: [P(E|H1), P(E|H2)]
        # the first likelihood is P(E|H1)=true positive rate=senitivity
        # the second likelihood is P(E|H2)=false positive rate=1-specificity
        self.likelihoods=likelihoods
        #boolean
        self.bayesian=bayesian


class Community:
    def __init__(self):
        self.agents = []
        self.subcommunities = {}
        self.rational_subcommunities = {}
        #by default we don't want to implement the doctrinal paradox
        self.doctrinal_paradox=False
        self.evidences=[]

    # initialize the community
    def initialize(self, N, epsilon, method, confidence_type,doctrinal_paradox):
        self.doctrinal_paradox=doctrinal_paradox
        #DP: here is the four way of generating an initial community of agent (with 3 opinions)
        if method == 'rationaly_random':
            for agent_index in range(N):
                x = random.random()
                y = random.random()
                z = LOGICAL_AND(x, y)
                self.agents.append(MultivariateAgent(agent_index, [x, y, z], epsilon,confidence_type,doctrinal_paradox=doctrinal_paradox))
        if method == 'purely_random':
            for agent_index in range(N):
                x = random.random()
                y = random.random()
                z = random.random()
                self.agents.append(MultivariateAgent(agent_index, [x, y, z], epsilon,confidence_type,doctrinal_paradox=doctrinal_paradox))
        if method == 'purely_random_excluding_irrationals':
            for agent_index in range(N):
                while True:
                    x = random.random()
                    y = random.random()
                    z = random.random()
                    if not IS_LOGICS_VIOLATED(x, y, z):
                        break
                self.agents.append(MultivariateAgent(agent_index, [x, y, z], epsilon,confidence_type, doctrinal_paradox=doctrinal_paradox))
        if method == 'uniform':
            for x_index in range(N):
                for y_index in range(N):
                    x = x_index / (N - 1)
                    y = y_index / (N - 1)
                    z = LOGICAL_AND(x, y)
                    self.agent.append(MultivariateAgent(1, [x, y, z], epsilon,confidence_type))
        #initializing a community of agents with one random opinion
        if method == '1D_random':
            self.agents = [ MultivariateAgent(agent_index, [random.random()], epsilon,confidence_type, doctrinal_paradox=doctrinal_paradox) for agent_index in range(N)]
        #the same but with opinion equaly spaning the whole opinion space
        if method == '1D_uniform':
            self.agents = [MultivariateAgent(agent_index, [agent_index/(N-1)], epsilon, confidence_type, doctrinal_paradox=doctrinal_paradox) for agent_index in range(N)]

    #this method updates all opinion of all agents untill the precision is reached (ie when the opinion stop changing significantly)
    def compute(self, precision=0.001):
        minimal_time=0
        #we introduced these lines beacause sometimes the precision is reached (the opinion are stable) but the piece of evidence is not yet available
        #because we are too early. We don't want to stop the script and continue the loop until reaching the piece of evidence time
        for evidence in self.evidences:
            if evidence.appearance_time>minimal_time:
                minimal_time=evidence.appearance_time +1
        time = 0
        is_precision_reached = False
        #continue updating opinion as long as the precision is not reached
        while not is_precision_reached :
            self.next_opinion(time)
            is_precision_reached = True
            for agent in self.agents:
                for opinion in agent.opinions:
                    if abs(opinion[-1] - opinion[-2]) > precision:
                        is_precision_reached = False
            if time < minimal_time :
                is_precision_reached = False
            time += 1

    #compute the next opinion
    def next_opinion(self, current_time):
        for agent in self.agents:
            agent.next_opinion(self, current_time)

    #DP: check is an agent hold irrational opinions
    def is_logics_violated(self):
        for agent in self.agents:
            if IS_LOGICS_VIOLATED(agent.opinions[0][-1], agent.opinions[1][-1], agent.opinions[2][-1]):
                return True
        return False

    #DP: compute the ratio of irrational agent in a community
    def irrational_agents_ratio(self):
        sum = 0
        for agent in self.agents:
            if IS_LOGICS_VIOLATED(agent.opinions[0][-1], agent.opinions[1][-1], agent.opinions[2][-1]):
                sum += 1
        return sum / len(self.agents)

    #DP: compute the ratio of agent who experienced irrationality in a community
    def experienced_irrationality_agents_ratio(self):
        sum = 0
        for agent in self.agents:
            if agent.has_experienced_irrationality:
                sum += 1
        return sum / len(self.agents)

    #DP: count how many subcommunities whe have a the end of the process
    def compute_subcommunities(self, precision):
        '''Two agents can be considered as belonging to the same subcommunity if their opinion distance is less than the confidence interval.
        Hence, we have considered that the universe is stabilized'''
        subcommunities = {}
        for agent in self.agents:
            found = False
            for key in subcommunities:
                if key[0] - precision < agent.opinions[0][-1] < key[0] + precision and key[1] - precision < \
                        agent.opinions[1][-1] < key[1] + precision and key[2] - precision < agent.opinions[2][-1] < key[
                    2] + precision:
                    subcommunities[key] += 1
                    found = True
                    break
            if not found:
                subcommunities[tuple([a[-1] for a in agent.opinions])] = 1
        self.subcommunities = subcommunities

    # DP: count how many rational subcommunities we have a the end of the process
    def compute_rational_subcommunities(self, precision):
        '''Two agents can be considered as belonging to the same subcommunity if their opinion distance is less than the confidence interval.
        Hence, we have considered that the universe is stabilized'''
        rational_subcommunities = {}
        for agent in self.agents:
            found = False
            for key in rational_subcommunities:
                if key[0] - precision < agent.opinions[0][-1] < key[0] + precision and key[1] - precision < \
                        agent.opinions[1][-1] < key[1] + precision and key[2] - precision < agent.opinions[2][-1] < key[
                    2] + precision and LOGICAL_AND(key[0], key[1]) - precision < LOGICAL_AND(agent.opinions[0][-1],
                                                                                             agent.opinions[1][
                                                                                                 -1]) < LOGICAL_AND(
                    key[0], key[1]) + precision:
                    rational_subcommunities[key] += 1
                    found = True
                    break
            if not found:
                rational_subcommunities[tuple([a[-1] for a in agent.opinions])] = 1
        self.rational_subcommunities = rational_subcommunities


# DP main functions (not relevant for Wouter)

def generate_atomic_multivariate_uniform_bound_confidence_model(N, epsilon, initialization_mode, confidence_type):
    community = Community()
    community.initialize(N, epsilon, initialization_mode, confidence_type, doctrinal_paradox=True)
    community.compute()
    return community


def simulate_montecarlo(N, epsilon, number_of_runs, initialization_mode, confidence_type):
    return [generate_atomic_multivariate_uniform_bound_confidence_model(N=N, epsilon=epsilon,
                                                                 initialization_mode=initialization_mode, confidence_type=confidence_type) for i in
            range(number_of_runs)]


def simulate_montecarlo_epsilon_variable(epsilon_start, epsilon_end, step, number_of_runs, N, initialization_mode,confidence_type):
    results = []
    start = timer()

    print(f'starting computations on {cpu_count()} cores')

    values = [[N, epsilon, number_of_runs, initialization_mode,confidence_type] for epsilon in np.arange(epsilon_start, epsilon_end, step)]

    with Pool() as pool:
        res = pool.starmap(simulate_montecarlo, values)


    end = timer()
    print(f'elapsed time: {end - start}')
    return res


def simulate_montecarlo_N_variable(N_start, N_end, step, number_of_runs, epsilon, initialization_mode):
    results = []
    for N in np.arange(N_start, N_end, step):
        print(N, end=" ")
        results.append(simulate_montecarlo(int(N), epsilon, number_of_runs, initialization_mode))
    return results


def display_world_line(community):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    for agent in community.agents:
        ax.plot(agent.opinions[0], agent.opinions[1], agent.opinions[2], label='parametric curve')
    plt.show()


def display_final_configuration(community):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for agent in community.agents:
        ax.scatter(agent.opinions[0][-1], agent.opinions[1][-1], agent.opinions[2][-1], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def display_scatter_plot(community):
    plt.figure(figsize=(5, 5))
    for agent in community.agents:
        plt.scatter(LOGICAL_AND(agent.opinions[0][-1], agent.opinions[1][-1]), agent.opinions[2][-1])
    plt.vlines(0.5, 0, 1, colors='k', linestyle='dotted')
    plt.hlines(0.5, 0, 1, colors='k', linestyle='dotted')
    plt.grid()
    plt.show()


def convert_into_binary(i):
    return 0 if i<0.5 else 1


def compare_MV_with_AB(n,N, epsilon,mode, confidence_type) :
    countMV=0
    countMVbin=0
    countAB=0
    mismatch=0

    for i in range(n):
        community = Community()
        community.initialize(N,epsilon,mode,confidence_type, doctrinal_paradox=True)
        x,y,z=0,0,0
        xbin,ybin,zbin=0,0,0
        for agent in community.agents:
            x+=agent.opinions[0][0]
            y+=agent.opinions[1][0]
            z+=agent.opinions[2][0]
            xbin+=convert_into_binary(agent.opinions[0][0])
            ybin+=convert_into_binary(agent.opinions[1][0])
            zbin+=convert_into_binary(agent.opinions[2][0] )
        MV = 1 if IS_LOGICS_VIOLATED(x/N,y/N,z/N) else 0
        countMV += MV
        countMVbin += 1 if  IS_LOGICS_VIOLATED(xbin/N,ybin/N,zbin/N) else 0
        community.compute()
        community.compute_subcommunities(0.05)
        x,y,z=0,0,0
        for agent in community.agents:
            x+=agent.opinions[0][-1]
            y+=agent.opinions[1][-1]
            z+=agent.opinions[2][-1]
        AB =  1 if IS_LOGICS_VIOLATED(x/N,y/N,z/N) else 0
        countAB += AB
        mismatch += 1 if AB+MV==1 else 0

    return [countMV/n, countMVbin/n, countAB/n, mismatch/n]



def main():

    sample_community = generate_atomic_multivariate_uniform_bound_confidence_model(25, 0.3,
                                                                            'purely_random_excluding_irrationals','linear')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    for agent in sample_community.agents:
        ax.plot(agent.opinions[0], agent.opinions[1], agent.opinions[2], label='parametric curve')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


# EBP main functions


#this function claculates a COLUMN of the heatmap. The 'x' coordinate is fixed and we vary vertically the 'y' one
def y_multi(x_name, y_name, x, y_range, consts):
    #we take the trivial linear case, like in HK's paper
    confidence_type="linear"
    #we set the alpha parameter to one, we consider only policy makers
    policy_maker_radicalisation_coefficient=1
    #list of the cummunities for a 'y' variable (and an 'x' fixed of course). Each of the element of this list is
    # another list called communities_y_fixed. The latter one is a list containing a list of communties having the same y and x.
    communities_y_variable=[]
    #loop over all possible values of 'y'
    for y in y_range[:-1]:
        communities_y_fixed=[]
        params=consts
        #setup the params with the value to compute. For instance params['epsilon']=0.3
        params[x_name]=x
        params[y_name] =y
        #creat 'n' communities with parameters 'x' and 'y'
        for community_id in range(0,params['n']):
            community = Community()
            community.agents=[]
            #create piece of evidence
            community.evidences=[Evidence([-0.],params['evidence_time'],[params['h1'],params['h2']])]
            #feeding the community with 'N' agents
            for i in range(params['N']):
                community.agents.append(MultivariateAgent(i, [random.random()], params['epsilon'], confidence_type, doctrinal_paradox=False, profession='policy-maker', self_radicalisation_coefficient=policy_maker_radicalisation_coefficient, tau=params['tau']))
            community.compute()
            #once ONE community is computed, add it to the communities_y_fixed list
            communities_y_fixed.append(community)
        #once ALL communities for the values 'x' and 'y' are computed, add it to the communities_y_variable list
        communities_y_variable.append(communities_y_fixed)

    #in each community, compute the ratio of agent convinced by the piece of evidence
    ratio_y_variable=[]
    # in each community, compute if the comminuty is consensual (1) or polarized (2)
    ratio_y_variable_polarisation = []

    for communities_y_fixed in communities_y_variable:
        number_positive=0
        total_number=0
        ratio_polarisation=0
        for community_y_fixed in communities_y_fixed:
            #compute the ration for ONE community with fixed x and y
            number_positive_this_community = 0
            for agent in community_y_fixed.agents:
                opinion= agent.opinions[0][-1]
                if opinion>0.5:
                    number_positive+=1
                    number_positive_this_community+=1
                total_number+=1
            #if ALL or NO ONE of the agent are positive, it means that we have a consensual society
            if number_positive_this_community==params['N'] or number_positive_this_community==0:
                ratio_polarisation+=1
            else :
                ratio_polarisation += 2

        ratio_y_variable.append(number_positive/total_number)
        ratio_y_variable_polarisation.append(ratio_polarisation/params['n'])
    #the function return are list of two items, each of them consisting in a list.
    return [ratio_y_variable, ratio_y_variable_polarisation]


#this fonction compute the cells of the heatmap for two variable parameters 'x' and 'y'
def x_vs_y_multi(x_name, y_name, x_range, y_range, consts ):
    #const is a dict with k-2 parameters among the k paramters N,n,epsilon, tau, evidence_time, h_1, h_2
    start = timer()
    print(f'starting computations on {cpu_count()} cores')
    #'value' is a list of list (!) of parameter. Each element of 'value' has a different value of 'x'. These elements will be given
    #to the algo for the multiprocessing
    values = [[x_name, y_name, x, y_range, consts] for x in x_range]
    #we run simultation in parallel. One for each value of 'x' in 'x_range'.
    with Pool() as pool:
        res = pool.starmap(y_multi, values)
    end = timer()
    print(f'elapsed time: {end - start}')
    #results 'res' are returned to the jupyter notebook and are ready for display. We just transpose the matrix.
    return [[res[j][i] for j in range(len(res))] for i in range(len(res[0]))]


#technical stuff for multiprocessing
if __name__ == '__main__':
    main()