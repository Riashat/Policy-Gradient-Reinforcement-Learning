function [strt_states,strt_actns,succ_states] = trajectory_data(trajectories)

d_state = length(trajectories.states(1,1,:)); %state dim
d_act = length(trajectories.actions(1,1,:)); %action dim
n_states = (length(trajectories.states(1,:,1))-1) * length(trajectories.states(:,1,1));

strt_states = trajectories.states(:,1:end-1,:);
strt_states = permute(strt_states,[2 1 3]); %just to keep states from the same trajectory together
strt_states = reshape(strt_states,n_states ,1, d_state);
strt_states = squeeze(strt_states);

strt_actns = trajectories.actions(:,1:end-1,:);
strt_actns = permute(strt_actns,[2 1 3]);
strt_actns = reshape(strt_actns,n_states ,1, d_act);
strt_actns = squeeze(strt_actns);

succ_states = trajectories.states(:,2:end,:);
succ_states = permute(succ_states,[2 1 3]);
succ_states = reshape(succ_states,n_states ,1, d_state);
succ_states = squeeze(succ_states);

