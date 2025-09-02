import tensorflow as tf


# Is it ready
# No


def rl_compiler(model, lr=1e-4,):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


@tf.function
def train_step(actor, critic, optimizer_actor, optimizer_critic,
               dynamics_model, raw_states, actions, advantages, returns,
               entropy_weight=0.01, training=True):
    # Get latent + weights from dynamics model (attention-based ensemble)
    model_out = dynamics_model(raw_states)  # {"latent": ..., "weight": ...}

    if training:
        with tf.GradientTape(persistent=True) as tape:
            # Actor forward pass
            logits = actor(model_out)  # planner logits
            action_probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)

            # Critic forward pass
            values = tf.squeeze(critic(model_out["latent"], actions), axis=-1)

            # Policy gradient loss
            indices = tf.stack([tf.range(tf.shape(logits)[0]), actions], axis=1)
            selected_log_probs = tf.gather_nd(log_probs, indices)

            policy_loss = -tf.reduce_mean(selected_log_probs * advantages)
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * log_probs, axis=1))
            actor_loss = policy_loss - entropy_weight * entropy

            # Value loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        # Backprop
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

        optimizer_actor.apply_gradients(zip(actor_grads, actor.trainable_variables))
        optimizer_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))
    else:
        logits = actor(model_out)
        entropy = tf.constant(0.0)
        actor_loss = tf.constant(0.0)
        critic_loss = tf.constant(0.0)

    action_probs = tf.nn.softmax(logits)
    planned_actions = tf.random.categorical(tf.math.log(action_probs), num_samples=1)
    planned_actions = tf.squeeze(planned_actions, axis=1)

    return actor_loss, critic_loss, entropy, planned_actions

