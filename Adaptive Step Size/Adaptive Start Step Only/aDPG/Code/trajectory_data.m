function [start_states,start_actions,successor_states] = trajectory_data(trajectories)

d_state = length(trajectories.states(1,1,:)); %state dim
d_act = length(trajectories.actions(1,1,:)); %action dim
n_states = (length(trajectories.states(1,:,1))-1) * length(trajectories.states(:,1,1));

start_states = trajectories.states(:,1:end-1,:);
start_states = permute(start_states,[2 1 3]); %just to keep states from the same trajectory together
start_states = reshape(start_states,n_states ,1, d_state);
start_states = squeeze(start_states);

start_actions = trajectories.actions(:,1:end-1,:);
start_actions = permute(start_actions,[2 1 3]);
start_actions = reshape(start_actions,n_states ,1, d_act);
start_actions = squeeze(start_actions);

successor_states = trajectories.states(:,2:end,:);
successor_states = permute(successor_states,[2 1 3]);
successor_states = reshape(successor_states,n_states ,1, d_state);
successor_states = squeeze(successor_states);

