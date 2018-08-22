import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np

mnist = np.load('mnist_scaled.npz')
X_train = mnist['X_train']
y_train = mnist['y_train']
X_test = mnist['X_test']
y_test = mnist['y_test']

mean_vals = np.mean(X_train,axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

'''g = tf.Graph()

with g.as_default():
	tf.set_random_seed(random_seed)
	tf_x = tf.placeholder(dtype=tf.float32,
		shape=(None,n_features),name='tf_x')
	tf_y = tf.placeholder(dtype=tf.int32,
		shape=None,name='tf_x')
	y_onehot = tf.one_hot(indices=tf_y,depth=n_classes)

	h1 = tf.layers.dense(inputs=tf_x,units=50,
		activation=tf.tanh,name='layer1')
	h2 = tf.layers.dense(inputs=h1,units=50,
		activation=tf.tanh,name='layer2')
	logits = tf.layers.dense(inputs=h2,units=10,
		activation=None,name='layer3')
	predictions = {
		'classes': tf.argmax(logits,axis=1),
		'probabilities': tf.nn.softmax(logits)
	}

with g.as_default():
	cost = tf.losses.softmax_cross_entropy(
		onehot_labels=y_onehot,logits=logits)
	optimizer = tf.train.GradientDescentOptimizer(
		learning_rate=0.001)
	train_op = optimizer.minimize(loss=cost)
	init_op = tf.global_variables_initializer()

def create_batch_generator(X,y,batch_size=128,shuffle=False):
	X_copy = np.array(X)
	y_copy = np.array(y)
	if shuffle:
		data = np.column_stack((X_copy,y_copy))
		np.random.shuffle(data)
		X_copy = data[:,:-1]
		y_copy = data[:,-1].astype(int)

	for i in range(0,X.shape[0],batch_size):
		yield(X_copy[i:i+batch_size,:],y_copy[i:i+batch_size])

sess = tf.Session(graph=g)
sess.run(init_op)

for epoch in range(10):
	training_costs = []
	batch_generator = create_batch_generator(
		X_train_centered,y_train,batch_size=64,shuffle=True)
	for batch_X,batch_y in batch_generator:
		feed = {tf_x:batch_X,tf_y:batch_y}
		aa,batch_cost = sess.run([train_op,cost],feed_dict=feed)
		training_costs.append(batch_cost)
	print(' -- Epoch %2d ''Avg. Training Loss: %.4f' % (
		epoch+1, np.mean(training_costs)))

feed = {tf_x:X_test_centered}
y_pred = sess.run(predictions['classes'],feed_dict=feed)
print('accuracy-',100*np.sum(y_pred==y_test)/y_test.shape[0])'''

y_train_onehot = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()

model.add(
	keras.layers.Dense(
		units=50,input_dim=X_train_centered.shape[1],
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',activation='tanh'))

model.add(
	keras.layers.Dense(
		units=50,input_dim=50,
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',activation='tanh'))

model.add(
	keras.layers.Dense(
		units=40,input_dim=50,
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',activation='tanh'))

model.add(
	keras.layers.Dense(
		units=y_train_onehot.shape[1],
		input_dim=40,kernel_initializer='glorot_uniform',
		bias_initializer='zeros',activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(
	lr=0.005,decay=1e-6,momentum=0.91)
model.compile(optimizer=sgd_optimizer,
	loss='categorical_crossentropy')

history = model.fit(X_train_centered,y_train_onehot,
	batch_size=64,epochs=5,verbose=1,validation_split=0.1)
