�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/dense_9983/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_9983/bias
}
*Adam/v/dense_9983/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9983/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_9983/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_9983/bias
}
*Adam/m/dense_9983/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9983/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_9983/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*)
shared_nameAdam/v/dense_9983/kernel
�
,Adam/v/dense_9983/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9983/kernel*
_output_shapes

:+*
dtype0
�
Adam/m/dense_9983/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*)
shared_nameAdam/m/dense_9983/kernel
�
,Adam/m/dense_9983/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9983/kernel*
_output_shapes

:+*
dtype0
�
Adam/v/dense_9982/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/v/dense_9982/bias
}
*Adam/v/dense_9982/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9982/bias*
_output_shapes
:+*
dtype0
�
Adam/m/dense_9982/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/m/dense_9982/bias
}
*Adam/m/dense_9982/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9982/bias*
_output_shapes
:+*
dtype0
�
Adam/v/dense_9982/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*)
shared_nameAdam/v/dense_9982/kernel
�
,Adam/v/dense_9982/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9982/kernel*
_output_shapes

:++*
dtype0
�
Adam/m/dense_9982/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*)
shared_nameAdam/m/dense_9982/kernel
�
,Adam/m/dense_9982/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9982/kernel*
_output_shapes

:++*
dtype0
�
Adam/v/dense_9981/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/v/dense_9981/bias
}
*Adam/v/dense_9981/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9981/bias*
_output_shapes
:+*
dtype0
�
Adam/m/dense_9981/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/m/dense_9981/bias
}
*Adam/m/dense_9981/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9981/bias*
_output_shapes
:+*
dtype0
�
Adam/v/dense_9981/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*)
shared_nameAdam/v/dense_9981/kernel
�
,Adam/v/dense_9981/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9981/kernel*
_output_shapes

:++*
dtype0
�
Adam/m/dense_9981/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*)
shared_nameAdam/m/dense_9981/kernel
�
,Adam/m/dense_9981/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9981/kernel*
_output_shapes

:++*
dtype0
�
Adam/v/dense_9980/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/v/dense_9980/bias
}
*Adam/v/dense_9980/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9980/bias*
_output_shapes
:+*
dtype0
�
Adam/m/dense_9980/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*'
shared_nameAdam/m/dense_9980/bias
}
*Adam/m/dense_9980/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9980/bias*
_output_shapes
:+*
dtype0
�
Adam/v/dense_9980/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*)
shared_nameAdam/v/dense_9980/kernel
�
,Adam/v/dense_9980/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9980/kernel*
_output_shapes

:+*
dtype0
�
Adam/m/dense_9980/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*)
shared_nameAdam/m/dense_9980/kernel
�
,Adam/m/dense_9980/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9980/kernel*
_output_shapes

:+*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
v
dense_9983/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_9983/bias
o
#dense_9983/bias/Read/ReadVariableOpReadVariableOpdense_9983/bias*
_output_shapes
:*
dtype0
~
dense_9983/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*"
shared_namedense_9983/kernel
w
%dense_9983/kernel/Read/ReadVariableOpReadVariableOpdense_9983/kernel*
_output_shapes

:+*
dtype0
v
dense_9982/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+* 
shared_namedense_9982/bias
o
#dense_9982/bias/Read/ReadVariableOpReadVariableOpdense_9982/bias*
_output_shapes
:+*
dtype0
~
dense_9982/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*"
shared_namedense_9982/kernel
w
%dense_9982/kernel/Read/ReadVariableOpReadVariableOpdense_9982/kernel*
_output_shapes

:++*
dtype0
v
dense_9981/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+* 
shared_namedense_9981/bias
o
#dense_9981/bias/Read/ReadVariableOpReadVariableOpdense_9981/bias*
_output_shapes
:+*
dtype0
~
dense_9981/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:++*"
shared_namedense_9981/kernel
w
%dense_9981/kernel/Read/ReadVariableOpReadVariableOpdense_9981/kernel*
_output_shapes

:++*
dtype0
v
dense_9980/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+* 
shared_namedense_9980/bias
o
#dense_9980/bias/Read/ReadVariableOpReadVariableOpdense_9980/bias*
_output_shapes
:+*
dtype0
~
dense_9980/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:+*"
shared_namedense_9980/kernel
w
%dense_9980/kernel/Read/ReadVariableOpReadVariableOpdense_9980/kernel*
_output_shapes

:+*
dtype0
�
 serving_default_dense_9980_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_9980_inputdense_9980/kerneldense_9980/biasdense_9981/kerneldense_9981/biasdense_9982/kerneldense_9982/biasdense_9983/kerneldense_9983/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_30815365

NoOpNoOp
�7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�6
value�6B�6 B�6
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*
* 
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

3trace_0
4trace_1* 

5trace_0
6trace_1* 
* 
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla*

>serving_default* 

0
1*

0
1*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
a[
VARIABLE_VALUEdense_9980/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_9980/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
a[
VARIABLE_VALUEdense_9981/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_9981/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
a[
VARIABLE_VALUEdense_9982/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_9982/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
a[
VARIABLE_VALUEdense_9983/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_9983/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

[0
\1*
* 
* 
* 
* 
* 
* 
�
80
]1
^2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12
i13
j14
k15
l16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
]0
_1
a2
c3
e4
g5
i6
k7*
<
^0
`1
b2
d3
f4
h5
j6
l7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
m	variables
n	keras_api
	ototal
	pcount*
t
q	variables
r	keras_api
strue_positives
ttrue_negatives
ufalse_positives
vfalse_negatives*
c]
VARIABLE_VALUEAdam/m/dense_9980/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_9980/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9980/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9980/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_9981/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_9981/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9981/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9981/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_9982/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_9982/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_9982/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_9982/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_9983/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_9983/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_9983/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_9983/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

m	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
 
s0
t1
u2
v3*

q	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_9980/kerneldense_9980/biasdense_9981/kerneldense_9981/biasdense_9982/kerneldense_9982/biasdense_9983/kerneldense_9983/bias	iterationlearning_rateAdam/m/dense_9980/kernelAdam/v/dense_9980/kernelAdam/m/dense_9980/biasAdam/v/dense_9980/biasAdam/m/dense_9981/kernelAdam/v/dense_9981/kernelAdam/m/dense_9981/biasAdam/v/dense_9981/biasAdam/m/dense_9982/kernelAdam/v/dense_9982/kernelAdam/m/dense_9982/biasAdam/v/dense_9982/biasAdam/m/dense_9983/kernelAdam/v/dense_9983/kernelAdam/m/dense_9983/biasAdam/v/dense_9983/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesConst*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_30815659
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9980/kerneldense_9980/biasdense_9981/kerneldense_9981/biasdense_9982/kerneldense_9982/biasdense_9983/kerneldense_9983/bias	iterationlearning_rateAdam/m/dense_9980/kernelAdam/v/dense_9980/kernelAdam/m/dense_9980/biasAdam/v/dense_9980/biasAdam/m/dense_9981/kernelAdam/v/dense_9981/kernelAdam/m/dense_9981/biasAdam/v/dense_9981/biasAdam/m/dense_9982/kernelAdam/v/dense_9982/kernelAdam/m/dense_9982/biasAdam/v/dense_9982/biasAdam/m/dense_9983/kernelAdam/v/dense_9983/kernelAdam/m/dense_9983/biasAdam/v/dense_9983/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_30815764َ
��
�
!__inference__traced_save_30815659
file_prefix:
(read_disablecopyonread_dense_9980_kernel:+6
(read_1_disablecopyonread_dense_9980_bias:+<
*read_2_disablecopyonread_dense_9981_kernel:++6
(read_3_disablecopyonread_dense_9981_bias:+<
*read_4_disablecopyonread_dense_9982_kernel:++6
(read_5_disablecopyonread_dense_9982_bias:+<
*read_6_disablecopyonread_dense_9983_kernel:+6
(read_7_disablecopyonread_dense_9983_bias:,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: D
2read_10_disablecopyonread_adam_m_dense_9980_kernel:+D
2read_11_disablecopyonread_adam_v_dense_9980_kernel:+>
0read_12_disablecopyonread_adam_m_dense_9980_bias:+>
0read_13_disablecopyonread_adam_v_dense_9980_bias:+D
2read_14_disablecopyonread_adam_m_dense_9981_kernel:++D
2read_15_disablecopyonread_adam_v_dense_9981_kernel:++>
0read_16_disablecopyonread_adam_m_dense_9981_bias:+>
0read_17_disablecopyonread_adam_v_dense_9981_bias:+D
2read_18_disablecopyonread_adam_m_dense_9982_kernel:++D
2read_19_disablecopyonread_adam_v_dense_9982_kernel:++>
0read_20_disablecopyonread_adam_m_dense_9982_bias:+>
0read_21_disablecopyonread_adam_v_dense_9982_bias:+D
2read_22_disablecopyonread_adam_m_dense_9983_kernel:+D
2read_23_disablecopyonread_adam_v_dense_9983_kernel:+>
0read_24_disablecopyonread_adam_m_dense_9983_bias:>
0read_25_disablecopyonread_adam_v_dense_9983_bias:)
read_26_disablecopyonread_total: )
read_27_disablecopyonread_count: 7
(read_28_disablecopyonread_true_positives:	�7
(read_29_disablecopyonread_true_negatives:	�8
)read_30_disablecopyonread_false_positives:	�8
)read_31_disablecopyonread_false_negatives:	�
savev2_const
identity_65��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_dense_9980_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_dense_9980_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:+|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_dense_9980_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_dense_9980_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:+~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_dense_9981_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_dense_9981_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:++|
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_9981_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_9981_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:+~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_dense_9982_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_dense_9982_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:++|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_9982_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_9982_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:+~
Read_6/DisableCopyOnReadDisableCopyOnRead*read_6_disablecopyonread_dense_9983_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp*read_6_disablecopyonread_dense_9983_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:+|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_9983_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_9983_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead2read_10_disablecopyonread_adam_m_dense_9980_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp2read_10_disablecopyonread_adam_m_dense_9980_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:+�
Read_11/DisableCopyOnReadDisableCopyOnRead2read_11_disablecopyonread_adam_v_dense_9980_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp2read_11_disablecopyonread_adam_v_dense_9980_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:+�
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_m_dense_9980_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_m_dense_9980_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_v_dense_9980_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_v_dense_9980_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_14/DisableCopyOnReadDisableCopyOnRead2read_14_disablecopyonread_adam_m_dense_9981_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp2read_14_disablecopyonread_adam_m_dense_9981_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:++�
Read_15/DisableCopyOnReadDisableCopyOnRead2read_15_disablecopyonread_adam_v_dense_9981_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp2read_15_disablecopyonread_adam_v_dense_9981_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:++�
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_m_dense_9981_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_m_dense_9981_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_v_dense_9981_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_v_dense_9981_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_18/DisableCopyOnReadDisableCopyOnRead2read_18_disablecopyonread_adam_m_dense_9982_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp2read_18_disablecopyonread_adam_m_dense_9982_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:++�
Read_19/DisableCopyOnReadDisableCopyOnRead2read_19_disablecopyonread_adam_v_dense_9982_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp2read_19_disablecopyonread_adam_v_dense_9982_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:++*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:++e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:++�
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_m_dense_9982_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_m_dense_9982_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_adam_v_dense_9982_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_adam_v_dense_9982_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:+*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:+a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:+�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_adam_m_dense_9983_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_adam_m_dense_9983_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:+�
Read_23/DisableCopyOnReadDisableCopyOnRead2read_23_disablecopyonread_adam_v_dense_9983_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp2read_23_disablecopyonread_adam_v_dense_9983_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:+*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:+e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:+�
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_m_dense_9983_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_m_dense_9983_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_v_dense_9983_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_v_dense_9983_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_total^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_27/DisableCopyOnReadDisableCopyOnReadread_27_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOpread_27_disablecopyonread_count^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_28/DisableCopyOnReadDisableCopyOnRead(read_28_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp(read_28_disablecopyonread_true_positives^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_29/DisableCopyOnReadDisableCopyOnRead(read_29_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp(read_29_disablecopyonread_true_negatives^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_false_positives^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_31/DisableCopyOnReadDisableCopyOnRead)read_31_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp)read_31_disablecopyonread_false_negatives^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 */
dtypes%
#2!	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_64Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_65IdentityIdentity_64:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_65Identity_65:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=!9

_output_shapes
: 

_user_specified_nameConst:/ +
)
_user_specified_namefalse_negatives:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_negatives:.*
(
_user_specified_nametrue_positives:%!

_user_specified_namecount:%!

_user_specified_nametotal:62
0
_user_specified_nameAdam/v/dense_9983/bias:62
0
_user_specified_nameAdam/m/dense_9983/bias:84
2
_user_specified_nameAdam/v/dense_9983/kernel:84
2
_user_specified_nameAdam/m/dense_9983/kernel:62
0
_user_specified_nameAdam/v/dense_9982/bias:62
0
_user_specified_nameAdam/m/dense_9982/bias:84
2
_user_specified_nameAdam/v/dense_9982/kernel:84
2
_user_specified_nameAdam/m/dense_9982/kernel:62
0
_user_specified_nameAdam/v/dense_9981/bias:62
0
_user_specified_nameAdam/m/dense_9981/bias:84
2
_user_specified_nameAdam/v/dense_9981/kernel:84
2
_user_specified_nameAdam/m/dense_9981/kernel:62
0
_user_specified_nameAdam/v/dense_9980/bias:62
0
_user_specified_nameAdam/m/dense_9980/bias:84
2
_user_specified_nameAdam/v/dense_9980/kernel:84
2
_user_specified_nameAdam/m/dense_9980/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:/+
)
_user_specified_namedense_9983/bias:1-
+
_user_specified_namedense_9983/kernel:/+
)
_user_specified_namedense_9982/bias:1-
+
_user_specified_namedense_9982/kernel:/+
)
_user_specified_namedense_9981/bias:1-
+
_user_specified_namedense_9981/kernel:/+
)
_user_specified_namedense_9980/bias:1-
+
_user_specified_namedense_9980/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_dense_9981_layer_call_fn_30815394

inputs
unknown:++
	unknown_0:+
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������+<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815390:($
"
_user_specified_name
30815388:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�

�
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815385

inputs0
matmul_readvariableop_resource:+-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:+*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_9983_layer_call_fn_30815434

inputs
unknown:+
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815430:($
"
_user_specified_name
30815428:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�

�
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815200

inputs0
matmul_readvariableop_resource:++-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:++*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�
�
-__inference_dense_9980_layer_call_fn_30815374

inputs
unknown:+
	unknown_0:+
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������+<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815370:($
"
_user_specified_name
30815368:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_9982_layer_call_fn_30815414

inputs
unknown:++
	unknown_0:+
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815216o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������+<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815410:($
"
_user_specified_name
30815408:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�
�
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815263
dense_9980_input%
dense_9980_30815242:+!
dense_9980_30815244:+%
dense_9981_30815247:++!
dense_9981_30815249:+%
dense_9982_30815252:++!
dense_9982_30815254:+%
dense_9983_30815257:+!
dense_9983_30815259:
identity��"dense_9980/StatefulPartitionedCall�"dense_9981/StatefulPartitionedCall�"dense_9982/StatefulPartitionedCall�"dense_9983/StatefulPartitionedCall�
"dense_9980/StatefulPartitionedCallStatefulPartitionedCalldense_9980_inputdense_9980_30815242dense_9980_30815244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815184�
"dense_9981/StatefulPartitionedCallStatefulPartitionedCall+dense_9980/StatefulPartitionedCall:output:0dense_9981_30815247dense_9981_30815249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815200�
"dense_9982/StatefulPartitionedCallStatefulPartitionedCall+dense_9981/StatefulPartitionedCall:output:0dense_9982_30815252dense_9982_30815254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815216�
"dense_9983/StatefulPartitionedCallStatefulPartitionedCall+dense_9982/StatefulPartitionedCall:output:0dense_9983_30815257dense_9983_30815259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815232z
IdentityIdentity+dense_9983/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_9980/StatefulPartitionedCall#^dense_9981/StatefulPartitionedCall#^dense_9982/StatefulPartitionedCall#^dense_9983/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2H
"dense_9980/StatefulPartitionedCall"dense_9980/StatefulPartitionedCall2H
"dense_9981/StatefulPartitionedCall"dense_9981/StatefulPartitionedCall2H
"dense_9982/StatefulPartitionedCall"dense_9982/StatefulPartitionedCall2H
"dense_9983/StatefulPartitionedCall"dense_9983/StatefulPartitionedCall:($
"
_user_specified_name
30815259:($
"
_user_specified_name
30815257:($
"
_user_specified_name
30815254:($
"
_user_specified_name
30815252:($
"
_user_specified_name
30815249:($
"
_user_specified_name
30815247:($
"
_user_specified_name
30815244:($
"
_user_specified_name
30815242:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input
�
�
&__inference_signature_wrapper_30815365
dense_9980_input
unknown:+
	unknown_0:+
	unknown_1:++
	unknown_2:+
	unknown_3:++
	unknown_4:+
	unknown_5:+
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_9980_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_30815171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815361:($
"
_user_specified_name
30815359:($
"
_user_specified_name
30815357:($
"
_user_specified_name
30815355:($
"
_user_specified_name
30815353:($
"
_user_specified_name
30815351:($
"
_user_specified_name
30815349:($
"
_user_specified_name
30815347:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input
�
�
2__inference_sequential_2500_layer_call_fn_30815305
dense_9980_input
unknown:+
	unknown_0:+
	unknown_1:++
	unknown_2:+
	unknown_3:++
	unknown_4:+
	unknown_5:+
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_9980_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815263o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815301:($
"
_user_specified_name
30815299:($
"
_user_specified_name
30815297:($
"
_user_specified_name
30815295:($
"
_user_specified_name
30815293:($
"
_user_specified_name
30815291:($
"
_user_specified_name
30815289:($
"
_user_specified_name
30815287:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input
�

�
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815216

inputs0
matmul_readvariableop_resource:++-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:++*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_30815764
file_prefix4
"assignvariableop_dense_9980_kernel:+0
"assignvariableop_1_dense_9980_bias:+6
$assignvariableop_2_dense_9981_kernel:++0
"assignvariableop_3_dense_9981_bias:+6
$assignvariableop_4_dense_9982_kernel:++0
"assignvariableop_5_dense_9982_bias:+6
$assignvariableop_6_dense_9983_kernel:+0
"assignvariableop_7_dense_9983_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: >
,assignvariableop_10_adam_m_dense_9980_kernel:+>
,assignvariableop_11_adam_v_dense_9980_kernel:+8
*assignvariableop_12_adam_m_dense_9980_bias:+8
*assignvariableop_13_adam_v_dense_9980_bias:+>
,assignvariableop_14_adam_m_dense_9981_kernel:++>
,assignvariableop_15_adam_v_dense_9981_kernel:++8
*assignvariableop_16_adam_m_dense_9981_bias:+8
*assignvariableop_17_adam_v_dense_9981_bias:+>
,assignvariableop_18_adam_m_dense_9982_kernel:++>
,assignvariableop_19_adam_v_dense_9982_kernel:++8
*assignvariableop_20_adam_m_dense_9982_bias:+8
*assignvariableop_21_adam_v_dense_9982_bias:+>
,assignvariableop_22_adam_m_dense_9983_kernel:+>
,assignvariableop_23_adam_v_dense_9983_kernel:+8
*assignvariableop_24_adam_m_dense_9983_bias:8
*assignvariableop_25_adam_v_dense_9983_bias:#
assignvariableop_26_total: #
assignvariableop_27_count: 1
"assignvariableop_28_true_positives:	�1
"assignvariableop_29_true_negatives:	�2
#assignvariableop_30_false_positives:	�2
#assignvariableop_31_false_negatives:	�
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_dense_9980_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_9980_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_9981_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_9981_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_9982_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_9982_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_9983_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_9983_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp,assignvariableop_10_adam_m_dense_9980_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_adam_v_dense_9980_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_m_dense_9980_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_v_dense_9980_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_m_dense_9981_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_v_dense_9981_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_m_dense_9981_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_v_dense_9981_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_m_dense_9982_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_v_dense_9982_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_m_dense_9982_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_v_dense_9982_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_m_dense_9983_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_v_dense_9983_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_m_dense_9983_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_v_dense_9983_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_true_positivesIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_true_negativesIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_false_positivesIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp#assignvariableop_31_false_negativesIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_33IdentityIdentity_32:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_33Identity_33:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:/ +
)
_user_specified_namefalse_negatives:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_negatives:.*
(
_user_specified_nametrue_positives:%!

_user_specified_namecount:%!

_user_specified_nametotal:62
0
_user_specified_nameAdam/v/dense_9983/bias:62
0
_user_specified_nameAdam/m/dense_9983/bias:84
2
_user_specified_nameAdam/v/dense_9983/kernel:84
2
_user_specified_nameAdam/m/dense_9983/kernel:62
0
_user_specified_nameAdam/v/dense_9982/bias:62
0
_user_specified_nameAdam/m/dense_9982/bias:84
2
_user_specified_nameAdam/v/dense_9982/kernel:84
2
_user_specified_nameAdam/m/dense_9982/kernel:62
0
_user_specified_nameAdam/v/dense_9981/bias:62
0
_user_specified_nameAdam/m/dense_9981/bias:84
2
_user_specified_nameAdam/v/dense_9981/kernel:84
2
_user_specified_nameAdam/m/dense_9981/kernel:62
0
_user_specified_nameAdam/v/dense_9980/bias:62
0
_user_specified_nameAdam/m/dense_9980/bias:84
2
_user_specified_nameAdam/v/dense_9980/kernel:84
2
_user_specified_nameAdam/m/dense_9980/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:/+
)
_user_specified_namedense_9983/bias:1-
+
_user_specified_namedense_9983/kernel:/+
)
_user_specified_namedense_9982/bias:1-
+
_user_specified_namedense_9982/kernel:/+
)
_user_specified_namedense_9981/bias:1-
+
_user_specified_namedense_9981/kernel:/+
)
_user_specified_namedense_9980/bias:1-
+
_user_specified_namedense_9980/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
2__inference_sequential_2500_layer_call_fn_30815284
dense_9980_input
unknown:+
	unknown_0:+
	unknown_1:++
	unknown_2:+
	unknown_3:++
	unknown_4:+
	unknown_5:+
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_9980_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815239o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
30815280:($
"
_user_specified_name
30815278:($
"
_user_specified_name
30815276:($
"
_user_specified_name
30815274:($
"
_user_specified_name
30815272:($
"
_user_specified_name
30815270:($
"
_user_specified_name
30815268:($
"
_user_specified_name
30815266:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input
�3
�
#__inference__wrapped_model_30815171
dense_9980_inputK
9sequential_2500_dense_9980_matmul_readvariableop_resource:+H
:sequential_2500_dense_9980_biasadd_readvariableop_resource:+K
9sequential_2500_dense_9981_matmul_readvariableop_resource:++H
:sequential_2500_dense_9981_biasadd_readvariableop_resource:+K
9sequential_2500_dense_9982_matmul_readvariableop_resource:++H
:sequential_2500_dense_9982_biasadd_readvariableop_resource:+K
9sequential_2500_dense_9983_matmul_readvariableop_resource:+H
:sequential_2500_dense_9983_biasadd_readvariableop_resource:
identity��1sequential_2500/dense_9980/BiasAdd/ReadVariableOp�0sequential_2500/dense_9980/MatMul/ReadVariableOp�1sequential_2500/dense_9981/BiasAdd/ReadVariableOp�0sequential_2500/dense_9981/MatMul/ReadVariableOp�1sequential_2500/dense_9982/BiasAdd/ReadVariableOp�0sequential_2500/dense_9982/MatMul/ReadVariableOp�1sequential_2500/dense_9983/BiasAdd/ReadVariableOp�0sequential_2500/dense_9983/MatMul/ReadVariableOp�
0sequential_2500/dense_9980/MatMul/ReadVariableOpReadVariableOp9sequential_2500_dense_9980_matmul_readvariableop_resource*
_output_shapes

:+*
dtype0�
!sequential_2500/dense_9980/MatMulMatMuldense_9980_input8sequential_2500/dense_9980/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
1sequential_2500/dense_9980/BiasAdd/ReadVariableOpReadVariableOp:sequential_2500_dense_9980_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype0�
"sequential_2500/dense_9980/BiasAddBiasAdd+sequential_2500/dense_9980/MatMul:product:09sequential_2500/dense_9980/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
sequential_2500/dense_9980/ReluRelu+sequential_2500/dense_9980/BiasAdd:output:0*
T0*'
_output_shapes
:���������+�
0sequential_2500/dense_9981/MatMul/ReadVariableOpReadVariableOp9sequential_2500_dense_9981_matmul_readvariableop_resource*
_output_shapes

:++*
dtype0�
!sequential_2500/dense_9981/MatMulMatMul-sequential_2500/dense_9980/Relu:activations:08sequential_2500/dense_9981/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
1sequential_2500/dense_9981/BiasAdd/ReadVariableOpReadVariableOp:sequential_2500_dense_9981_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype0�
"sequential_2500/dense_9981/BiasAddBiasAdd+sequential_2500/dense_9981/MatMul:product:09sequential_2500/dense_9981/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
sequential_2500/dense_9981/ReluRelu+sequential_2500/dense_9981/BiasAdd:output:0*
T0*'
_output_shapes
:���������+�
0sequential_2500/dense_9982/MatMul/ReadVariableOpReadVariableOp9sequential_2500_dense_9982_matmul_readvariableop_resource*
_output_shapes

:++*
dtype0�
!sequential_2500/dense_9982/MatMulMatMul-sequential_2500/dense_9981/Relu:activations:08sequential_2500/dense_9982/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
1sequential_2500/dense_9982/BiasAdd/ReadVariableOpReadVariableOp:sequential_2500_dense_9982_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype0�
"sequential_2500/dense_9982/BiasAddBiasAdd+sequential_2500/dense_9982/MatMul:product:09sequential_2500/dense_9982/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+�
sequential_2500/dense_9982/ReluRelu+sequential_2500/dense_9982/BiasAdd:output:0*
T0*'
_output_shapes
:���������+�
0sequential_2500/dense_9983/MatMul/ReadVariableOpReadVariableOp9sequential_2500_dense_9983_matmul_readvariableop_resource*
_output_shapes

:+*
dtype0�
!sequential_2500/dense_9983/MatMulMatMul-sequential_2500/dense_9982/Relu:activations:08sequential_2500/dense_9983/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1sequential_2500/dense_9983/BiasAdd/ReadVariableOpReadVariableOp:sequential_2500_dense_9983_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"sequential_2500/dense_9983/BiasAddBiasAdd+sequential_2500/dense_9983/MatMul:product:09sequential_2500/dense_9983/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"sequential_2500/dense_9983/SigmoidSigmoid+sequential_2500/dense_9983/BiasAdd:output:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_2500/dense_9983/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp2^sequential_2500/dense_9980/BiasAdd/ReadVariableOp1^sequential_2500/dense_9980/MatMul/ReadVariableOp2^sequential_2500/dense_9981/BiasAdd/ReadVariableOp1^sequential_2500/dense_9981/MatMul/ReadVariableOp2^sequential_2500/dense_9982/BiasAdd/ReadVariableOp1^sequential_2500/dense_9982/MatMul/ReadVariableOp2^sequential_2500/dense_9983/BiasAdd/ReadVariableOp1^sequential_2500/dense_9983/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2f
1sequential_2500/dense_9980/BiasAdd/ReadVariableOp1sequential_2500/dense_9980/BiasAdd/ReadVariableOp2d
0sequential_2500/dense_9980/MatMul/ReadVariableOp0sequential_2500/dense_9980/MatMul/ReadVariableOp2f
1sequential_2500/dense_9981/BiasAdd/ReadVariableOp1sequential_2500/dense_9981/BiasAdd/ReadVariableOp2d
0sequential_2500/dense_9981/MatMul/ReadVariableOp0sequential_2500/dense_9981/MatMul/ReadVariableOp2f
1sequential_2500/dense_9982/BiasAdd/ReadVariableOp1sequential_2500/dense_9982/BiasAdd/ReadVariableOp2d
0sequential_2500/dense_9982/MatMul/ReadVariableOp0sequential_2500/dense_9982/MatMul/ReadVariableOp2f
1sequential_2500/dense_9983/BiasAdd/ReadVariableOp1sequential_2500/dense_9983/BiasAdd/ReadVariableOp2d
0sequential_2500/dense_9983/MatMul/ReadVariableOp0sequential_2500/dense_9983/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input
�

�
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815425

inputs0
matmul_readvariableop_resource:++-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:++*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�

�
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815405

inputs0
matmul_readvariableop_resource:++-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:++*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�

�
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815445

inputs0
matmul_readvariableop_resource:+-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:+*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�

�
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815184

inputs0
matmul_readvariableop_resource:+-
biasadd_readvariableop_resource:+
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:+*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������+P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������+a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������+S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815232

inputs0
matmul_readvariableop_resource:+-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:+*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������+: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������+
 
_user_specified_nameinputs
�
�
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815239
dense_9980_input%
dense_9980_30815185:+!
dense_9980_30815187:+%
dense_9981_30815201:++!
dense_9981_30815203:+%
dense_9982_30815217:++!
dense_9982_30815219:+%
dense_9983_30815233:+!
dense_9983_30815235:
identity��"dense_9980/StatefulPartitionedCall�"dense_9981/StatefulPartitionedCall�"dense_9982/StatefulPartitionedCall�"dense_9983/StatefulPartitionedCall�
"dense_9980/StatefulPartitionedCallStatefulPartitionedCalldense_9980_inputdense_9980_30815185dense_9980_30815187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815184�
"dense_9981/StatefulPartitionedCallStatefulPartitionedCall+dense_9980/StatefulPartitionedCall:output:0dense_9981_30815201dense_9981_30815203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815200�
"dense_9982/StatefulPartitionedCallStatefulPartitionedCall+dense_9981/StatefulPartitionedCall:output:0dense_9982_30815217dense_9982_30815219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815216�
"dense_9983/StatefulPartitionedCallStatefulPartitionedCall+dense_9982/StatefulPartitionedCall:output:0dense_9983_30815233dense_9983_30815235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815232z
IdentityIdentity+dense_9983/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^dense_9980/StatefulPartitionedCall#^dense_9981/StatefulPartitionedCall#^dense_9982/StatefulPartitionedCall#^dense_9983/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2H
"dense_9980/StatefulPartitionedCall"dense_9980/StatefulPartitionedCall2H
"dense_9981/StatefulPartitionedCall"dense_9981/StatefulPartitionedCall2H
"dense_9982/StatefulPartitionedCall"dense_9982/StatefulPartitionedCall2H
"dense_9983/StatefulPartitionedCall"dense_9983/StatefulPartitionedCall:($
"
_user_specified_name
30815235:($
"
_user_specified_name
30815233:($
"
_user_specified_name
30815219:($
"
_user_specified_name
30815217:($
"
_user_specified_name
30815203:($
"
_user_specified_name
30815201:($
"
_user_specified_name
30815187:($
"
_user_specified_name
30815185:Y U
'
_output_shapes
:���������
*
_user_specified_namedense_9980_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
dense_9980_input9
"serving_default_dense_9980_input:0���������>

dense_99830
StatefulPartitionedCall:0���������tensorflow/serving/predict:�v
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
3trace_0
4trace_12�
2__inference_sequential_2500_layer_call_fn_30815284
2__inference_sequential_2500_layer_call_fn_30815305�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3trace_0z4trace_1
�
5trace_0
6trace_12�
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815239
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815263�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z5trace_0z6trace_1
�B�
#__inference__wrapped_model_30815171dense_9980_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla"
experimentalOptimizer
,
>serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
-__inference_dense_9980_layer_call_fn_30815374�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
�
Etrace_02�
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815385�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
#:!+2dense_9980/kernel
:+2dense_9980/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_02�
-__inference_dense_9981_layer_call_fn_30815394�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
�
Ltrace_02�
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815405�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
#:!++2dense_9981/kernel
:+2dense_9981/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_02�
-__inference_dense_9982_layer_call_fn_30815414�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
�
Strace_02�
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815425�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
#:!++2dense_9982/kernel
:+2dense_9982/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_02�
-__inference_dense_9983_layer_call_fn_30815434�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
�
Ztrace_02�
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815445�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
#:!+2dense_9983/kernel
:2dense_9983/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2500_layer_call_fn_30815284dense_9980_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_sequential_2500_layer_call_fn_30815305dense_9980_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815239dense_9980_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815263dense_9980_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
80
]1
^2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12
i13
j14
k15
l16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
]0
_1
a2
c3
e4
g5
i6
k7"
trackable_list_wrapper
X
^0
`1
b2
d3
f4
h5
j6
l7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_30815365dense_9980_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_9980_layer_call_fn_30815374inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815385inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_9981_layer_call_fn_30815394inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815405inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_9982_layer_call_fn_30815414inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815425inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_9983_layer_call_fn_30815434inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815445inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
m	variables
n	keras_api
	ototal
	pcount"
_tf_keras_metric
�
q	variables
r	keras_api
strue_positives
ttrue_negatives
ufalse_positives
vfalse_negatives"
_tf_keras_metric
(:&+2Adam/m/dense_9980/kernel
(:&+2Adam/v/dense_9980/kernel
": +2Adam/m/dense_9980/bias
": +2Adam/v/dense_9980/bias
(:&++2Adam/m/dense_9981/kernel
(:&++2Adam/v/dense_9981/kernel
": +2Adam/m/dense_9981/bias
": +2Adam/v/dense_9981/bias
(:&++2Adam/m/dense_9982/kernel
(:&++2Adam/v/dense_9982/kernel
": +2Adam/m/dense_9982/bias
": +2Adam/v/dense_9982/bias
(:&+2Adam/m/dense_9983/kernel
(:&+2Adam/v/dense_9983/kernel
": 2Adam/m/dense_9983/bias
": 2Adam/v/dense_9983/bias
.
o0
p1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
<
s0
t1
u2
v3"
trackable_list_wrapper
-
q	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives�
#__inference__wrapped_model_30815171~$%,-9�6
/�,
*�'
dense_9980_input���������
� "7�4
2

dense_9983$�!

dense_9983����������
H__inference_dense_9980_layer_call_and_return_conditional_losses_30815385c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������+
� �
-__inference_dense_9980_layer_call_fn_30815374X/�,
%�"
 �
inputs���������
� "!�
unknown���������+�
H__inference_dense_9981_layer_call_and_return_conditional_losses_30815405c/�,
%�"
 �
inputs���������+
� ",�)
"�
tensor_0���������+
� �
-__inference_dense_9981_layer_call_fn_30815394X/�,
%�"
 �
inputs���������+
� "!�
unknown���������+�
H__inference_dense_9982_layer_call_and_return_conditional_losses_30815425c$%/�,
%�"
 �
inputs���������+
� ",�)
"�
tensor_0���������+
� �
-__inference_dense_9982_layer_call_fn_30815414X$%/�,
%�"
 �
inputs���������+
� "!�
unknown���������+�
H__inference_dense_9983_layer_call_and_return_conditional_losses_30815445c,-/�,
%�"
 �
inputs���������+
� ",�)
"�
tensor_0���������
� �
-__inference_dense_9983_layer_call_fn_30815434X,-/�,
%�"
 �
inputs���������+
� "!�
unknown����������
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815239{$%,-A�>
7�4
*�'
dense_9980_input���������
p

 
� ",�)
"�
tensor_0���������
� �
M__inference_sequential_2500_layer_call_and_return_conditional_losses_30815263{$%,-A�>
7�4
*�'
dense_9980_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
2__inference_sequential_2500_layer_call_fn_30815284p$%,-A�>
7�4
*�'
dense_9980_input���������
p

 
� "!�
unknown����������
2__inference_sequential_2500_layer_call_fn_30815305p$%,-A�>
7�4
*�'
dense_9980_input���������
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_30815365�$%,-M�J
� 
C�@
>
dense_9980_input*�'
dense_9980_input���������"7�4
2

dense_9983$�!

dense_9983���������