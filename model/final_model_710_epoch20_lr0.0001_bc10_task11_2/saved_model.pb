??6
?
?

B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02unknown8??.
?
conv1d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv1d_35/kernel
y
$conv1d_35/kernel/Read/ReadVariableOpReadVariableOpconv1d_35/kernel*"
_output_shapes
:@*
dtype0
t
conv1d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_35/bias
m
"conv1d_35/bias/Read/ReadVariableOpReadVariableOpconv1d_35/bias*
_output_shapes
:@*
dtype0
?
conv1d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv1d_36/kernel
z
$conv1d_36/kernel/Read/ReadVariableOpReadVariableOpconv1d_36/kernel*#
_output_shapes
:@?*
dtype0
u
conv1d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_36/bias
n
"conv1d_36/bias/Read/ReadVariableOpReadVariableOpconv1d_36/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_20/gamma
?
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_20/beta
?
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_20/moving_mean
?
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_20/moving_variance
?
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes	
:?*
dtype0
?
conv1d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv1d_37/kernel
{
$conv1d_37/kernel/Read/ReadVariableOpReadVariableOpconv1d_37/kernel*$
_output_shapes
:??*
dtype0
u
conv1d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_37/bias
n
"conv1d_37/bias/Read/ReadVariableOpReadVariableOpconv1d_37/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes	
:?*
dtype0
?
conv1d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv1d_38/kernel
{
$conv1d_38/kernel/Read/ReadVariableOpReadVariableOpconv1d_38/kernel*$
_output_shapes
:??*
dtype0
u
conv1d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_38/bias
n
"conv1d_38/bias/Read/ReadVariableOpReadVariableOpconv1d_38/bias*
_output_shapes	
:?*
dtype0
?
conv1d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv1d_39/kernel
{
$conv1d_39/kernel/Read/ReadVariableOpReadVariableOpconv1d_39/kernel*$
_output_shapes
:??*
dtype0
u
conv1d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_39/bias
n
"conv1d_39/bias/Read/ReadVariableOpReadVariableOpconv1d_39/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_22/gamma
?
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_22/beta
?
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_22/moving_mean
?
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_22/moving_variance
?
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:?*
dtype0
?
conv1d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv1d_40/kernel
{
$conv1d_40/kernel/Read/ReadVariableOpReadVariableOpconv1d_40/kernel*$
_output_shapes
:??*
dtype0
u
conv1d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_40/bias
n
"conv1d_40/bias/Read/ReadVariableOpReadVariableOpconv1d_40/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:?*
dtype0
?
conv1d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv1d_41/kernel
{
$conv1d_41/kernel/Read/ReadVariableOpReadVariableOpconv1d_41/kernel*$
_output_shapes
:??*
dtype0
u
conv1d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d_41/bias
n
"conv1d_41/bias/Read/ReadVariableOpReadVariableOpconv1d_41/bias*
_output_shapes	
:?*
dtype0
?
one_dropout1_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameone_dropout1_5/kernel
?
)one_dropout1_5/kernel/Read/ReadVariableOpReadVariableOpone_dropout1_5/kernel*
_output_shapes
:	?@*
dtype0
~
one_dropout1_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameone_dropout1_5/bias
w
'one_dropout1_5/bias/Read/ReadVariableOpReadVariableOpone_dropout1_5/bias*
_output_shapes
:@*
dtype0
?
second_dropout1_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*)
shared_namesecond_dropout1_5/kernel
?
,second_dropout1_5/kernel/Read/ReadVariableOpReadVariableOpsecond_dropout1_5/kernel*
_output_shapes
:	?@*
dtype0
?
second_dropout1_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namesecond_dropout1_5/bias
}
*second_dropout1_5/bias/Read/ReadVariableOpReadVariableOpsecond_dropout1_5/bias*
_output_shapes
:@*
dtype0
?
one_dropout2_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*&
shared_nameone_dropout2_5/kernel
?
)one_dropout2_5/kernel/Read/ReadVariableOpReadVariableOpone_dropout2_5/kernel*
_output_shapes
:	?A*
dtype0
~
one_dropout2_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameone_dropout2_5/bias
w
'one_dropout2_5/bias/Read/ReadVariableOpReadVariableOpone_dropout2_5/bias*
_output_shapes
:*
dtype0
?
second_dropout2_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*)
shared_namesecond_dropout2_5/kernel
?
,second_dropout2_5/kernel/Read/ReadVariableOpReadVariableOpsecond_dropout2_5/kernel*
_output_shapes
:	?A*
dtype0
?
second_dropout2_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namesecond_dropout2_5/bias
}
*second_dropout2_5/bias/Read/ReadVariableOpReadVariableOpsecond_dropout2_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
?
Adam/conv1d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_35/kernel/m
?
+Adam/conv1d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/kernel/m*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_35/bias/m
{
)Adam/conv1d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv1d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv1d_36/kernel/m
?
+Adam/conv1d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/m*#
_output_shapes
:@?*
dtype0
?
Adam/conv1d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_36/bias/m
|
)Adam/conv1d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_20/gamma/m
?
7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_20/beta/m
?
6Adam/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_37/kernel/m
?
+Adam/conv1d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_37/bias/m
|
)Adam/conv1d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_21/gamma/m
?
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_21/beta/m
?
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_38/kernel/m
?
+Adam/conv1d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_38/bias/m
|
)Adam/conv1d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_39/kernel/m
?
+Adam/conv1d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_39/bias/m
|
)Adam/conv1d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_22/gamma/m
?
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_22/beta/m
?
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_40/kernel/m
?
+Adam/conv1d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_40/bias/m
|
)Adam/conv1d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_23/gamma/m
?
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_23/beta/m
?
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_41/kernel/m
?
+Adam/conv1d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/kernel/m*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_41/bias/m
|
)Adam/conv1d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/one_dropout1_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_nameAdam/one_dropout1_5/kernel/m
?
0Adam/one_dropout1_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/one_dropout1_5/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/one_dropout1_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/one_dropout1_5/bias/m
?
.Adam/one_dropout1_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/one_dropout1_5/bias/m*
_output_shapes
:@*
dtype0
?
Adam/second_dropout1_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*0
shared_name!Adam/second_dropout1_5/kernel/m
?
3Adam/second_dropout1_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/second_dropout1_5/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/second_dropout1_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/second_dropout1_5/bias/m
?
1Adam/second_dropout1_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/second_dropout1_5/bias/m*
_output_shapes
:@*
dtype0
?
Adam/one_dropout2_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*-
shared_nameAdam/one_dropout2_5/kernel/m
?
0Adam/one_dropout2_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/one_dropout2_5/kernel/m*
_output_shapes
:	?A*
dtype0
?
Adam/one_dropout2_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/one_dropout2_5/bias/m
?
.Adam/one_dropout2_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/one_dropout2_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/second_dropout2_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*0
shared_name!Adam/second_dropout2_5/kernel/m
?
3Adam/second_dropout2_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/second_dropout2_5/kernel/m*
_output_shapes
:	?A*
dtype0
?
Adam/second_dropout2_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/second_dropout2_5/bias/m
?
1Adam/second_dropout2_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/second_dropout2_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv1d_35/kernel/v
?
+Adam/conv1d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/kernel/v*"
_output_shapes
:@*
dtype0
?
Adam/conv1d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv1d_35/bias/v
{
)Adam/conv1d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_35/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv1d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv1d_36/kernel/v
?
+Adam/conv1d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/kernel/v*#
_output_shapes
:@?*
dtype0
?
Adam/conv1d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_36/bias/v
|
)Adam/conv1d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_36/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_20/gamma/v
?
7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_20/beta/v
?
6Adam/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_37/kernel/v
?
+Adam/conv1d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_37/bias/v
|
)Adam/conv1d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_37/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_21/gamma/v
?
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_21/beta/v
?
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_38/kernel/v
?
+Adam/conv1d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_38/bias/v
|
)Adam/conv1d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_38/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_39/kernel/v
?
+Adam/conv1d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_39/bias/v
|
)Adam/conv1d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_39/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_22/gamma/v
?
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_22/beta/v
?
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_40/kernel/v
?
+Adam/conv1d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_40/bias/v
|
)Adam/conv1d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_40/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_23/gamma/v
?
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_23/beta/v
?
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv1d_41/kernel/v
?
+Adam/conv1d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/kernel/v*$
_output_shapes
:??*
dtype0
?
Adam/conv1d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv1d_41/bias/v
|
)Adam/conv1d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_41/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/one_dropout1_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_nameAdam/one_dropout1_5/kernel/v
?
0Adam/one_dropout1_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/one_dropout1_5/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/one_dropout1_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/one_dropout1_5/bias/v
?
.Adam/one_dropout1_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/one_dropout1_5/bias/v*
_output_shapes
:@*
dtype0
?
Adam/second_dropout1_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*0
shared_name!Adam/second_dropout1_5/kernel/v
?
3Adam/second_dropout1_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/second_dropout1_5/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/second_dropout1_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/second_dropout1_5/bias/v
?
1Adam/second_dropout1_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/second_dropout1_5/bias/v*
_output_shapes
:@*
dtype0
?
Adam/one_dropout2_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*-
shared_nameAdam/one_dropout2_5/kernel/v
?
0Adam/one_dropout2_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/one_dropout2_5/kernel/v*
_output_shapes
:	?A*
dtype0
?
Adam/one_dropout2_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/one_dropout2_5/bias/v
?
.Adam/one_dropout2_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/one_dropout2_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/second_dropout2_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?A*0
shared_name!Adam/second_dropout2_5/kernel/v
?
3Adam/second_dropout2_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/second_dropout2_5/kernel/v*
_output_shapes
:	?A*
dtype0
?
Adam/second_dropout2_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/second_dropout2_5/bias/v
?
1Adam/second_dropout2_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/second_dropout2_5/bias/v*
_output_shapes
:*
dtype0
O
ConstConst*
_output_shapes
:*
dtype0*
valueB:

NoOpNoOp
??
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-11
layer-24
layer_with_weights-12
layer-25
layer-26
layer-27
layer_with_weights-13
layer-28
layer_with_weights-14
layer-29
layer-30
 layer-31
!	optimizer
"loss
#	variables
$regularization_losses
%trainable_variables
&	keras_api
'
signatures
 
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
R
.	variables
/regularization_losses
0trainable_variables
1	keras_api
h

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
R
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
h

Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
?
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
R
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
h

\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
R
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
h

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
?
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
R
y	variables
zregularization_losses
{trainable_variables
|	keras_api
k

}kernel
~bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate(m?)m?2m?3m?=m?>m?Im?Jm?Pm?Qm?\m?]m?fm?gm?qm?rm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?(v?)v?2v?3v?=v?>v?Iv?Jv?Pv?Qv?\v?]v?fv?gv?qv?rv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
(0
)1
22
33
=4
>5
?6
@7
I8
J9
P10
Q11
R12
S13
\14
]15
f16
g17
q18
r19
s20
t21
}22
~23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
 
?
(0
)1
22
33
=4
>5
I6
J7
P8
Q9
\10
]11
f12
g13
q14
r15
}16
~17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?
#	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
$regularization_losses
?non_trainable_variables
%trainable_variables
 
\Z
VARIABLE_VALUEconv1d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
?layers
*	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
+regularization_losses
?non_trainable_variables
,trainable_variables
 
 
 
?
?layers
.	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
/regularization_losses
?non_trainable_variables
0trainable_variables
\Z
VARIABLE_VALUEconv1d_36/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_36/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
?
?layers
4	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
5regularization_losses
?non_trainable_variables
6trainable_variables
 
 
 
?
?layers
8	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
9regularization_losses
?non_trainable_variables
:trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
?2
@3
 

=0
>1
?
?layers
A	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Bregularization_losses
?non_trainable_variables
Ctrainable_variables
 
 
 
?
?layers
E	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Fregularization_losses
?non_trainable_variables
Gtrainable_variables
\Z
VARIABLE_VALUEconv1d_37/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_37/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
?
?layers
K	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Lregularization_losses
?non_trainable_variables
Mtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
R2
S3
 

P0
Q1
?
?layers
T	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Uregularization_losses
?non_trainable_variables
Vtrainable_variables
 
 
 
?
?layers
X	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
\Z
VARIABLE_VALUEconv1d_38/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_38/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1
 

\0
]1
?
?layers
^	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
_regularization_losses
?non_trainable_variables
`trainable_variables
 
 
 
?
?layers
b	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
cregularization_losses
?non_trainable_variables
dtrainable_variables
\Z
VARIABLE_VALUEconv1d_39/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_39/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
?
?layers
h	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
iregularization_losses
?non_trainable_variables
jtrainable_variables
 
 
 
?
?layers
l	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
mregularization_losses
?non_trainable_variables
ntrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
s2
t3
 

q0
r1
?
?layers
u	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
vregularization_losses
?non_trainable_variables
wtrainable_variables
 
 
 
?
?layers
y	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
zregularization_losses
?non_trainable_variables
{trainable_variables
\Z
VARIABLE_VALUEconv1d_40/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_40/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1
 

}0
~1
?
?layers
	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
][
VARIABLE_VALUEconv1d_41/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv1d_41/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
b`
VARIABLE_VALUEone_dropout1_5/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEone_dropout1_5/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
ec
VARIABLE_VALUEsecond_dropout1_5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsecond_dropout1_5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
b`
VARIABLE_VALUEone_dropout2_5/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEone_dropout2_5/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
ec
VARIABLE_VALUEsecond_dropout2_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsecond_dropout2_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
 
 
 
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
 
 
Y
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
:
?0
@1
R2
S3
s4
t5
?6
?7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

R0
S1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

s0
t1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv1d_35/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_35/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_36/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_20/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_38/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_40/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_40/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv1d_41/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_41/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/one_dropout1_5/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/one_dropout1_5/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout1_5/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout1_5/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/one_dropout2_5/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/one_dropout2_5/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout2_5/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout2_5/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_35/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_35/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_36/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_36/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_20/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_37/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_37/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_38/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_38/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_39/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_39/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv1d_40/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1d_40/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv1d_41/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d_41/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/one_dropout1_5/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/one_dropout1_5/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout1_5/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout1_5/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/one_dropout2_5/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/one_dropout2_5/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout2_5/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/second_dropout2_5/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_6Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6conv1d_35/kernelconv1d_35/biasconv1d_36/kernelconv1d_36/bias&batch_normalization_20/moving_variancebatch_normalization_20/gamma"batch_normalization_20/moving_meanbatch_normalization_20/betaconv1d_37/kernelconv1d_37/bias&batch_normalization_21/moving_variancebatch_normalization_21/gamma"batch_normalization_21/moving_meanbatch_normalization_21/betaconv1d_38/kernelconv1d_38/biasconv1d_39/kernelconv1d_39/bias&batch_normalization_22/moving_variancebatch_normalization_22/gamma"batch_normalization_22/moving_meanbatch_normalization_22/betaConstconv1d_40/kernelconv1d_40/bias&batch_normalization_23/moving_variancebatch_normalization_23/gamma"batch_normalization_23/moving_meanbatch_normalization_23/betaconv1d_41/kernelconv1d_41/biassecond_dropout1_5/kernelsecond_dropout1_5/biasone_dropout1_5/kernelone_dropout1_5/biassecond_dropout2_5/kernelsecond_dropout2_5/biasone_dropout2_5/kernelone_dropout2_5/bias*3
Tin,
*2(*
Tout
2*L
_output_shapes:
8:??????????????????:??????????????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_49604082
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
?
Const_3Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_eb4b8947c9234bfe95159d1f6d0e9bfa/part
h
SelectSelectStaticRegexFullMatchConst_2Const_3"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?C
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:}*
dtype0*?C
value?CB?C}B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:}*
dtype0*?
value?B?}B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?,
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices$conv1d_35/kernel/Read/ReadVariableOp"conv1d_35/bias/Read/ReadVariableOp$conv1d_36/kernel/Read/ReadVariableOp"conv1d_36/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv1d_37/kernel/Read/ReadVariableOp"conv1d_37/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv1d_38/kernel/Read/ReadVariableOp"conv1d_38/bias/Read/ReadVariableOp$conv1d_39/kernel/Read/ReadVariableOp"conv1d_39/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv1d_40/kernel/Read/ReadVariableOp"conv1d_40/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp$conv1d_41/kernel/Read/ReadVariableOp"conv1d_41/bias/Read/ReadVariableOp)one_dropout1_5/kernel/Read/ReadVariableOp'one_dropout1_5/bias/Read/ReadVariableOp,second_dropout1_5/kernel/Read/ReadVariableOp*second_dropout1_5/bias/Read/ReadVariableOp)one_dropout2_5/kernel/Read/ReadVariableOp'one_dropout2_5/bias/Read/ReadVariableOp,second_dropout2_5/kernel/Read/ReadVariableOp*second_dropout2_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOp+Adam/conv1d_35/kernel/m/Read/ReadVariableOp)Adam/conv1d_35/bias/m/Read/ReadVariableOp+Adam/conv1d_36/kernel/m/Read/ReadVariableOp)Adam/conv1d_36/bias/m/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_20/beta/m/Read/ReadVariableOp+Adam/conv1d_37/kernel/m/Read/ReadVariableOp)Adam/conv1d_37/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp+Adam/conv1d_38/kernel/m/Read/ReadVariableOp)Adam/conv1d_38/bias/m/Read/ReadVariableOp+Adam/conv1d_39/kernel/m/Read/ReadVariableOp)Adam/conv1d_39/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp+Adam/conv1d_40/kernel/m/Read/ReadVariableOp)Adam/conv1d_40/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp+Adam/conv1d_41/kernel/m/Read/ReadVariableOp)Adam/conv1d_41/bias/m/Read/ReadVariableOp0Adam/one_dropout1_5/kernel/m/Read/ReadVariableOp.Adam/one_dropout1_5/bias/m/Read/ReadVariableOp3Adam/second_dropout1_5/kernel/m/Read/ReadVariableOp1Adam/second_dropout1_5/bias/m/Read/ReadVariableOp0Adam/one_dropout2_5/kernel/m/Read/ReadVariableOp.Adam/one_dropout2_5/bias/m/Read/ReadVariableOp3Adam/second_dropout2_5/kernel/m/Read/ReadVariableOp1Adam/second_dropout2_5/bias/m/Read/ReadVariableOp+Adam/conv1d_35/kernel/v/Read/ReadVariableOp)Adam/conv1d_35/bias/v/Read/ReadVariableOp+Adam/conv1d_36/kernel/v/Read/ReadVariableOp)Adam/conv1d_36/bias/v/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_20/beta/v/Read/ReadVariableOp+Adam/conv1d_37/kernel/v/Read/ReadVariableOp)Adam/conv1d_37/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp+Adam/conv1d_38/kernel/v/Read/ReadVariableOp)Adam/conv1d_38/bias/v/Read/ReadVariableOp+Adam/conv1d_39/kernel/v/Read/ReadVariableOp)Adam/conv1d_39/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp+Adam/conv1d_40/kernel/v/Read/ReadVariableOp)Adam/conv1d_40/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp+Adam/conv1d_41/kernel/v/Read/ReadVariableOp)Adam/conv1d_41/bias/v/Read/ReadVariableOp0Adam/one_dropout1_5/kernel/v/Read/ReadVariableOp.Adam/one_dropout1_5/bias/v/Read/ReadVariableOp3Adam/second_dropout1_5/kernel/v/Read/ReadVariableOp1Adam/second_dropout1_5/bias/v/Read/ReadVariableOp0Adam/one_dropout2_5/kernel/v/Read/ReadVariableOp.Adam/one_dropout2_5/bias/v/Read/ReadVariableOp3Adam/second_dropout2_5/kernel/v/Read/ReadVariableOp1Adam/second_dropout2_5/bias/v/Read/ReadVariableOp"/device:CPU:0*?
dtypes?
2}	
h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
?
SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst_1"/device:CPU:0*
dtypes
2
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?C
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:}*
dtype0*?C
value?CB?C}B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:}*
dtype0*?
value?B?}B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
2}	
D

Identity_1Identity	RestoreV2*
T0*
_output_shapes
:
O
AssignVariableOpAssignVariableOpconv1d_35/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:1*
T0*
_output_shapes
:
O
AssignVariableOp_1AssignVariableOpconv1d_35/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:2*
T0*
_output_shapes
:
Q
AssignVariableOp_2AssignVariableOpconv1d_36/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpconv1d_36/bias
Identity_4*
dtype0
F

Identity_5IdentityRestoreV2:4*
T0*
_output_shapes
:
]
AssignVariableOp_4AssignVariableOpbatch_normalization_20/gamma
Identity_5*
dtype0
F

Identity_6IdentityRestoreV2:5*
T0*
_output_shapes
:
\
AssignVariableOp_5AssignVariableOpbatch_normalization_20/beta
Identity_6*
dtype0
F

Identity_7IdentityRestoreV2:6*
T0*
_output_shapes
:
c
AssignVariableOp_6AssignVariableOp"batch_normalization_20/moving_mean
Identity_7*
dtype0
F

Identity_8IdentityRestoreV2:7*
T0*
_output_shapes
:
g
AssignVariableOp_7AssignVariableOp&batch_normalization_20/moving_variance
Identity_8*
dtype0
F

Identity_9IdentityRestoreV2:8*
T0*
_output_shapes
:
Q
AssignVariableOp_8AssignVariableOpconv1d_37/kernel
Identity_9*
dtype0
G
Identity_10IdentityRestoreV2:9*
T0*
_output_shapes
:
P
AssignVariableOp_9AssignVariableOpconv1d_37/biasIdentity_10*
dtype0
H
Identity_11IdentityRestoreV2:10*
T0*
_output_shapes
:
_
AssignVariableOp_10AssignVariableOpbatch_normalization_21/gammaIdentity_11*
dtype0
H
Identity_12IdentityRestoreV2:11*
T0*
_output_shapes
:
^
AssignVariableOp_11AssignVariableOpbatch_normalization_21/betaIdentity_12*
dtype0
H
Identity_13IdentityRestoreV2:12*
T0*
_output_shapes
:
e
AssignVariableOp_12AssignVariableOp"batch_normalization_21/moving_meanIdentity_13*
dtype0
H
Identity_14IdentityRestoreV2:13*
T0*
_output_shapes
:
i
AssignVariableOp_13AssignVariableOp&batch_normalization_21/moving_varianceIdentity_14*
dtype0
H
Identity_15IdentityRestoreV2:14*
T0*
_output_shapes
:
S
AssignVariableOp_14AssignVariableOpconv1d_38/kernelIdentity_15*
dtype0
H
Identity_16IdentityRestoreV2:15*
T0*
_output_shapes
:
Q
AssignVariableOp_15AssignVariableOpconv1d_38/biasIdentity_16*
dtype0
H
Identity_17IdentityRestoreV2:16*
T0*
_output_shapes
:
S
AssignVariableOp_16AssignVariableOpconv1d_39/kernelIdentity_17*
dtype0
H
Identity_18IdentityRestoreV2:17*
T0*
_output_shapes
:
Q
AssignVariableOp_17AssignVariableOpconv1d_39/biasIdentity_18*
dtype0
H
Identity_19IdentityRestoreV2:18*
T0*
_output_shapes
:
_
AssignVariableOp_18AssignVariableOpbatch_normalization_22/gammaIdentity_19*
dtype0
H
Identity_20IdentityRestoreV2:19*
T0*
_output_shapes
:
^
AssignVariableOp_19AssignVariableOpbatch_normalization_22/betaIdentity_20*
dtype0
H
Identity_21IdentityRestoreV2:20*
T0*
_output_shapes
:
e
AssignVariableOp_20AssignVariableOp"batch_normalization_22/moving_meanIdentity_21*
dtype0
H
Identity_22IdentityRestoreV2:21*
T0*
_output_shapes
:
i
AssignVariableOp_21AssignVariableOp&batch_normalization_22/moving_varianceIdentity_22*
dtype0
H
Identity_23IdentityRestoreV2:22*
T0*
_output_shapes
:
S
AssignVariableOp_22AssignVariableOpconv1d_40/kernelIdentity_23*
dtype0
H
Identity_24IdentityRestoreV2:23*
T0*
_output_shapes
:
Q
AssignVariableOp_23AssignVariableOpconv1d_40/biasIdentity_24*
dtype0
H
Identity_25IdentityRestoreV2:24*
T0*
_output_shapes
:
_
AssignVariableOp_24AssignVariableOpbatch_normalization_23/gammaIdentity_25*
dtype0
H
Identity_26IdentityRestoreV2:25*
T0*
_output_shapes
:
^
AssignVariableOp_25AssignVariableOpbatch_normalization_23/betaIdentity_26*
dtype0
H
Identity_27IdentityRestoreV2:26*
T0*
_output_shapes
:
e
AssignVariableOp_26AssignVariableOp"batch_normalization_23/moving_meanIdentity_27*
dtype0
H
Identity_28IdentityRestoreV2:27*
T0*
_output_shapes
:
i
AssignVariableOp_27AssignVariableOp&batch_normalization_23/moving_varianceIdentity_28*
dtype0
H
Identity_29IdentityRestoreV2:28*
T0*
_output_shapes
:
S
AssignVariableOp_28AssignVariableOpconv1d_41/kernelIdentity_29*
dtype0
H
Identity_30IdentityRestoreV2:29*
T0*
_output_shapes
:
Q
AssignVariableOp_29AssignVariableOpconv1d_41/biasIdentity_30*
dtype0
H
Identity_31IdentityRestoreV2:30*
T0*
_output_shapes
:
X
AssignVariableOp_30AssignVariableOpone_dropout1_5/kernelIdentity_31*
dtype0
H
Identity_32IdentityRestoreV2:31*
T0*
_output_shapes
:
V
AssignVariableOp_31AssignVariableOpone_dropout1_5/biasIdentity_32*
dtype0
H
Identity_33IdentityRestoreV2:32*
T0*
_output_shapes
:
[
AssignVariableOp_32AssignVariableOpsecond_dropout1_5/kernelIdentity_33*
dtype0
H
Identity_34IdentityRestoreV2:33*
T0*
_output_shapes
:
Y
AssignVariableOp_33AssignVariableOpsecond_dropout1_5/biasIdentity_34*
dtype0
H
Identity_35IdentityRestoreV2:34*
T0*
_output_shapes
:
X
AssignVariableOp_34AssignVariableOpone_dropout2_5/kernelIdentity_35*
dtype0
H
Identity_36IdentityRestoreV2:35*
T0*
_output_shapes
:
V
AssignVariableOp_35AssignVariableOpone_dropout2_5/biasIdentity_36*
dtype0
H
Identity_37IdentityRestoreV2:36*
T0*
_output_shapes
:
[
AssignVariableOp_36AssignVariableOpsecond_dropout2_5/kernelIdentity_37*
dtype0
H
Identity_38IdentityRestoreV2:37*
T0*
_output_shapes
:
Y
AssignVariableOp_37AssignVariableOpsecond_dropout2_5/biasIdentity_38*
dtype0
H
Identity_39IdentityRestoreV2:38*
T0	*
_output_shapes
:
L
AssignVariableOp_38AssignVariableOp	Adam/iterIdentity_39*
dtype0	
H
Identity_40IdentityRestoreV2:39*
T0*
_output_shapes
:
N
AssignVariableOp_39AssignVariableOpAdam/beta_1Identity_40*
dtype0
H
Identity_41IdentityRestoreV2:40*
T0*
_output_shapes
:
N
AssignVariableOp_40AssignVariableOpAdam/beta_2Identity_41*
dtype0
H
Identity_42IdentityRestoreV2:41*
T0*
_output_shapes
:
M
AssignVariableOp_41AssignVariableOp
Adam/decayIdentity_42*
dtype0
H
Identity_43IdentityRestoreV2:42*
T0*
_output_shapes
:
U
AssignVariableOp_42AssignVariableOpAdam/learning_rateIdentity_43*
dtype0
H
Identity_44IdentityRestoreV2:43*
T0*
_output_shapes
:
H
AssignVariableOp_43AssignVariableOptotalIdentity_44*
dtype0
H
Identity_45IdentityRestoreV2:44*
T0*
_output_shapes
:
H
AssignVariableOp_44AssignVariableOpcountIdentity_45*
dtype0
H
Identity_46IdentityRestoreV2:45*
T0*
_output_shapes
:
J
AssignVariableOp_45AssignVariableOptotal_1Identity_46*
dtype0
H
Identity_47IdentityRestoreV2:46*
T0*
_output_shapes
:
J
AssignVariableOp_46AssignVariableOpcount_1Identity_47*
dtype0
H
Identity_48IdentityRestoreV2:47*
T0*
_output_shapes
:
J
AssignVariableOp_47AssignVariableOptotal_2Identity_48*
dtype0
H
Identity_49IdentityRestoreV2:48*
T0*
_output_shapes
:
J
AssignVariableOp_48AssignVariableOpcount_2Identity_49*
dtype0
H
Identity_50IdentityRestoreV2:49*
T0*
_output_shapes
:
J
AssignVariableOp_49AssignVariableOptotal_3Identity_50*
dtype0
H
Identity_51IdentityRestoreV2:50*
T0*
_output_shapes
:
J
AssignVariableOp_50AssignVariableOpcount_3Identity_51*
dtype0
H
Identity_52IdentityRestoreV2:51*
T0*
_output_shapes
:
J
AssignVariableOp_51AssignVariableOptotal_4Identity_52*
dtype0
H
Identity_53IdentityRestoreV2:52*
T0*
_output_shapes
:
J
AssignVariableOp_52AssignVariableOpcount_4Identity_53*
dtype0
H
Identity_54IdentityRestoreV2:53*
T0*
_output_shapes
:
J
AssignVariableOp_53AssignVariableOptotal_5Identity_54*
dtype0
H
Identity_55IdentityRestoreV2:54*
T0*
_output_shapes
:
J
AssignVariableOp_54AssignVariableOpcount_5Identity_55*
dtype0
H
Identity_56IdentityRestoreV2:55*
T0*
_output_shapes
:
J
AssignVariableOp_55AssignVariableOptotal_6Identity_56*
dtype0
H
Identity_57IdentityRestoreV2:56*
T0*
_output_shapes
:
J
AssignVariableOp_56AssignVariableOpcount_6Identity_57*
dtype0
H
Identity_58IdentityRestoreV2:57*
T0*
_output_shapes
:
J
AssignVariableOp_57AssignVariableOptotal_7Identity_58*
dtype0
H
Identity_59IdentityRestoreV2:58*
T0*
_output_shapes
:
J
AssignVariableOp_58AssignVariableOpcount_7Identity_59*
dtype0
H
Identity_60IdentityRestoreV2:59*
T0*
_output_shapes
:
J
AssignVariableOp_59AssignVariableOptotal_8Identity_60*
dtype0
H
Identity_61IdentityRestoreV2:60*
T0*
_output_shapes
:
J
AssignVariableOp_60AssignVariableOpcount_8Identity_61*
dtype0
H
Identity_62IdentityRestoreV2:61*
T0*
_output_shapes
:
J
AssignVariableOp_61AssignVariableOptotal_9Identity_62*
dtype0
H
Identity_63IdentityRestoreV2:62*
T0*
_output_shapes
:
J
AssignVariableOp_62AssignVariableOpcount_9Identity_63*
dtype0
H
Identity_64IdentityRestoreV2:63*
T0*
_output_shapes
:
K
AssignVariableOp_63AssignVariableOptotal_10Identity_64*
dtype0
H
Identity_65IdentityRestoreV2:64*
T0*
_output_shapes
:
K
AssignVariableOp_64AssignVariableOpcount_10Identity_65*
dtype0
H
Identity_66IdentityRestoreV2:65*
T0*
_output_shapes
:
Z
AssignVariableOp_65AssignVariableOpAdam/conv1d_35/kernel/mIdentity_66*
dtype0
H
Identity_67IdentityRestoreV2:66*
T0*
_output_shapes
:
X
AssignVariableOp_66AssignVariableOpAdam/conv1d_35/bias/mIdentity_67*
dtype0
H
Identity_68IdentityRestoreV2:67*
T0*
_output_shapes
:
Z
AssignVariableOp_67AssignVariableOpAdam/conv1d_36/kernel/mIdentity_68*
dtype0
H
Identity_69IdentityRestoreV2:68*
T0*
_output_shapes
:
X
AssignVariableOp_68AssignVariableOpAdam/conv1d_36/bias/mIdentity_69*
dtype0
H
Identity_70IdentityRestoreV2:69*
T0*
_output_shapes
:
f
AssignVariableOp_69AssignVariableOp#Adam/batch_normalization_20/gamma/mIdentity_70*
dtype0
H
Identity_71IdentityRestoreV2:70*
T0*
_output_shapes
:
e
AssignVariableOp_70AssignVariableOp"Adam/batch_normalization_20/beta/mIdentity_71*
dtype0
H
Identity_72IdentityRestoreV2:71*
T0*
_output_shapes
:
Z
AssignVariableOp_71AssignVariableOpAdam/conv1d_37/kernel/mIdentity_72*
dtype0
H
Identity_73IdentityRestoreV2:72*
T0*
_output_shapes
:
X
AssignVariableOp_72AssignVariableOpAdam/conv1d_37/bias/mIdentity_73*
dtype0
H
Identity_74IdentityRestoreV2:73*
T0*
_output_shapes
:
f
AssignVariableOp_73AssignVariableOp#Adam/batch_normalization_21/gamma/mIdentity_74*
dtype0
H
Identity_75IdentityRestoreV2:74*
T0*
_output_shapes
:
e
AssignVariableOp_74AssignVariableOp"Adam/batch_normalization_21/beta/mIdentity_75*
dtype0
H
Identity_76IdentityRestoreV2:75*
T0*
_output_shapes
:
Z
AssignVariableOp_75AssignVariableOpAdam/conv1d_38/kernel/mIdentity_76*
dtype0
H
Identity_77IdentityRestoreV2:76*
T0*
_output_shapes
:
X
AssignVariableOp_76AssignVariableOpAdam/conv1d_38/bias/mIdentity_77*
dtype0
H
Identity_78IdentityRestoreV2:77*
T0*
_output_shapes
:
Z
AssignVariableOp_77AssignVariableOpAdam/conv1d_39/kernel/mIdentity_78*
dtype0
H
Identity_79IdentityRestoreV2:78*
T0*
_output_shapes
:
X
AssignVariableOp_78AssignVariableOpAdam/conv1d_39/bias/mIdentity_79*
dtype0
H
Identity_80IdentityRestoreV2:79*
T0*
_output_shapes
:
f
AssignVariableOp_79AssignVariableOp#Adam/batch_normalization_22/gamma/mIdentity_80*
dtype0
H
Identity_81IdentityRestoreV2:80*
T0*
_output_shapes
:
e
AssignVariableOp_80AssignVariableOp"Adam/batch_normalization_22/beta/mIdentity_81*
dtype0
H
Identity_82IdentityRestoreV2:81*
T0*
_output_shapes
:
Z
AssignVariableOp_81AssignVariableOpAdam/conv1d_40/kernel/mIdentity_82*
dtype0
H
Identity_83IdentityRestoreV2:82*
T0*
_output_shapes
:
X
AssignVariableOp_82AssignVariableOpAdam/conv1d_40/bias/mIdentity_83*
dtype0
H
Identity_84IdentityRestoreV2:83*
T0*
_output_shapes
:
f
AssignVariableOp_83AssignVariableOp#Adam/batch_normalization_23/gamma/mIdentity_84*
dtype0
H
Identity_85IdentityRestoreV2:84*
T0*
_output_shapes
:
e
AssignVariableOp_84AssignVariableOp"Adam/batch_normalization_23/beta/mIdentity_85*
dtype0
H
Identity_86IdentityRestoreV2:85*
T0*
_output_shapes
:
Z
AssignVariableOp_85AssignVariableOpAdam/conv1d_41/kernel/mIdentity_86*
dtype0
H
Identity_87IdentityRestoreV2:86*
T0*
_output_shapes
:
X
AssignVariableOp_86AssignVariableOpAdam/conv1d_41/bias/mIdentity_87*
dtype0
H
Identity_88IdentityRestoreV2:87*
T0*
_output_shapes
:
_
AssignVariableOp_87AssignVariableOpAdam/one_dropout1_5/kernel/mIdentity_88*
dtype0
H
Identity_89IdentityRestoreV2:88*
T0*
_output_shapes
:
]
AssignVariableOp_88AssignVariableOpAdam/one_dropout1_5/bias/mIdentity_89*
dtype0
H
Identity_90IdentityRestoreV2:89*
T0*
_output_shapes
:
b
AssignVariableOp_89AssignVariableOpAdam/second_dropout1_5/kernel/mIdentity_90*
dtype0
H
Identity_91IdentityRestoreV2:90*
T0*
_output_shapes
:
`
AssignVariableOp_90AssignVariableOpAdam/second_dropout1_5/bias/mIdentity_91*
dtype0
H
Identity_92IdentityRestoreV2:91*
T0*
_output_shapes
:
_
AssignVariableOp_91AssignVariableOpAdam/one_dropout2_5/kernel/mIdentity_92*
dtype0
H
Identity_93IdentityRestoreV2:92*
T0*
_output_shapes
:
]
AssignVariableOp_92AssignVariableOpAdam/one_dropout2_5/bias/mIdentity_93*
dtype0
H
Identity_94IdentityRestoreV2:93*
T0*
_output_shapes
:
b
AssignVariableOp_93AssignVariableOpAdam/second_dropout2_5/kernel/mIdentity_94*
dtype0
H
Identity_95IdentityRestoreV2:94*
T0*
_output_shapes
:
`
AssignVariableOp_94AssignVariableOpAdam/second_dropout2_5/bias/mIdentity_95*
dtype0
H
Identity_96IdentityRestoreV2:95*
T0*
_output_shapes
:
Z
AssignVariableOp_95AssignVariableOpAdam/conv1d_35/kernel/vIdentity_96*
dtype0
H
Identity_97IdentityRestoreV2:96*
T0*
_output_shapes
:
X
AssignVariableOp_96AssignVariableOpAdam/conv1d_35/bias/vIdentity_97*
dtype0
H
Identity_98IdentityRestoreV2:97*
T0*
_output_shapes
:
Z
AssignVariableOp_97AssignVariableOpAdam/conv1d_36/kernel/vIdentity_98*
dtype0
H
Identity_99IdentityRestoreV2:98*
T0*
_output_shapes
:
X
AssignVariableOp_98AssignVariableOpAdam/conv1d_36/bias/vIdentity_99*
dtype0
I
Identity_100IdentityRestoreV2:99*
T0*
_output_shapes
:
g
AssignVariableOp_99AssignVariableOp#Adam/batch_normalization_20/gamma/vIdentity_100*
dtype0
J
Identity_101IdentityRestoreV2:100*
T0*
_output_shapes
:
g
AssignVariableOp_100AssignVariableOp"Adam/batch_normalization_20/beta/vIdentity_101*
dtype0
J
Identity_102IdentityRestoreV2:101*
T0*
_output_shapes
:
\
AssignVariableOp_101AssignVariableOpAdam/conv1d_37/kernel/vIdentity_102*
dtype0
J
Identity_103IdentityRestoreV2:102*
T0*
_output_shapes
:
Z
AssignVariableOp_102AssignVariableOpAdam/conv1d_37/bias/vIdentity_103*
dtype0
J
Identity_104IdentityRestoreV2:103*
T0*
_output_shapes
:
h
AssignVariableOp_103AssignVariableOp#Adam/batch_normalization_21/gamma/vIdentity_104*
dtype0
J
Identity_105IdentityRestoreV2:104*
T0*
_output_shapes
:
g
AssignVariableOp_104AssignVariableOp"Adam/batch_normalization_21/beta/vIdentity_105*
dtype0
J
Identity_106IdentityRestoreV2:105*
T0*
_output_shapes
:
\
AssignVariableOp_105AssignVariableOpAdam/conv1d_38/kernel/vIdentity_106*
dtype0
J
Identity_107IdentityRestoreV2:106*
T0*
_output_shapes
:
Z
AssignVariableOp_106AssignVariableOpAdam/conv1d_38/bias/vIdentity_107*
dtype0
J
Identity_108IdentityRestoreV2:107*
T0*
_output_shapes
:
\
AssignVariableOp_107AssignVariableOpAdam/conv1d_39/kernel/vIdentity_108*
dtype0
J
Identity_109IdentityRestoreV2:108*
T0*
_output_shapes
:
Z
AssignVariableOp_108AssignVariableOpAdam/conv1d_39/bias/vIdentity_109*
dtype0
J
Identity_110IdentityRestoreV2:109*
T0*
_output_shapes
:
h
AssignVariableOp_109AssignVariableOp#Adam/batch_normalization_22/gamma/vIdentity_110*
dtype0
J
Identity_111IdentityRestoreV2:110*
T0*
_output_shapes
:
g
AssignVariableOp_110AssignVariableOp"Adam/batch_normalization_22/beta/vIdentity_111*
dtype0
J
Identity_112IdentityRestoreV2:111*
T0*
_output_shapes
:
\
AssignVariableOp_111AssignVariableOpAdam/conv1d_40/kernel/vIdentity_112*
dtype0
J
Identity_113IdentityRestoreV2:112*
T0*
_output_shapes
:
Z
AssignVariableOp_112AssignVariableOpAdam/conv1d_40/bias/vIdentity_113*
dtype0
J
Identity_114IdentityRestoreV2:113*
T0*
_output_shapes
:
h
AssignVariableOp_113AssignVariableOp#Adam/batch_normalization_23/gamma/vIdentity_114*
dtype0
J
Identity_115IdentityRestoreV2:114*
T0*
_output_shapes
:
g
AssignVariableOp_114AssignVariableOp"Adam/batch_normalization_23/beta/vIdentity_115*
dtype0
J
Identity_116IdentityRestoreV2:115*
T0*
_output_shapes
:
\
AssignVariableOp_115AssignVariableOpAdam/conv1d_41/kernel/vIdentity_116*
dtype0
J
Identity_117IdentityRestoreV2:116*
T0*
_output_shapes
:
Z
AssignVariableOp_116AssignVariableOpAdam/conv1d_41/bias/vIdentity_117*
dtype0
J
Identity_118IdentityRestoreV2:117*
T0*
_output_shapes
:
a
AssignVariableOp_117AssignVariableOpAdam/one_dropout1_5/kernel/vIdentity_118*
dtype0
J
Identity_119IdentityRestoreV2:118*
T0*
_output_shapes
:
_
AssignVariableOp_118AssignVariableOpAdam/one_dropout1_5/bias/vIdentity_119*
dtype0
J
Identity_120IdentityRestoreV2:119*
T0*
_output_shapes
:
d
AssignVariableOp_119AssignVariableOpAdam/second_dropout1_5/kernel/vIdentity_120*
dtype0
J
Identity_121IdentityRestoreV2:120*
T0*
_output_shapes
:
b
AssignVariableOp_120AssignVariableOpAdam/second_dropout1_5/bias/vIdentity_121*
dtype0
J
Identity_122IdentityRestoreV2:121*
T0*
_output_shapes
:
a
AssignVariableOp_121AssignVariableOpAdam/one_dropout2_5/kernel/vIdentity_122*
dtype0
J
Identity_123IdentityRestoreV2:122*
T0*
_output_shapes
:
_
AssignVariableOp_122AssignVariableOpAdam/one_dropout2_5/bias/vIdentity_123*
dtype0
J
Identity_124IdentityRestoreV2:123*
T0*
_output_shapes
:
d
AssignVariableOp_123AssignVariableOpAdam/second_dropout2_5/kernel/vIdentity_124*
dtype0
J
Identity_125IdentityRestoreV2:124*
T0*
_output_shapes
:
b
AssignVariableOp_124AssignVariableOpAdam/second_dropout2_5/bias/vIdentity_125*
dtype0
?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
?
RestoreV2_1	RestoreV2saver_filenameRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2

NoOp_1NoOp"/device:CPU:0
?
Identity_126Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??'
?
?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604564

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
*__inference_model_5_layer_call_fn_49603403
input_69
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource3
/batch_normalization_20_assignmovingavg_496029675
1batch_normalization_20_assignmovingavg_1_49602973@
<batch_normalization_20_batchnorm_mul_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource3
/batch_normalization_21_assignmovingavg_496030195
1batch_normalization_21_assignmovingavg_1_49603025@
<batch_normalization_21_batchnorm_mul_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource3
/batch_normalization_22_assignmovingavg_496030885
1batch_normalization_22_assignmovingavg_1_49603094@
<batch_normalization_22_batchnorm_mul_readvariableop_resource<
8batch_normalization_22_batchnorm_readvariableop_resource:
6conv1d_40_required_space_to_batch_paddings_block_shape9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource3
/batch_normalization_23_assignmovingavg_496031515
1batch_normalization_23_assignmovingavg_1_49603157@
<batch_normalization_23_batchnorm_mul_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource5
1second_dropout1_tensordot_readvariableop_resource3
/second_dropout1_biasadd_readvariableop_resource2
.one_dropout1_tensordot_readvariableop_resource0
,one_dropout1_biasadd_readvariableop_resource5
1second_dropout2_tensordot_readvariableop_resource3
/second_dropout2_biasadd_readvariableop_resource2
.one_dropout2_tensordot_readvariableop_resource0
,one_dropout2_biasadd_readvariableop_resource
identity

identity_1??:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsinput_6(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1dConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/BiasAdd{
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/Relu?
max_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_15/ExpandDims/dim?
max_pooling1d_15/ExpandDims
ExpandDimsconv1d_35/Relu:activations:0(max_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_15/ExpandDims?
max_pooling1d_15/MaxPoolMaxPool$max_pooling1d_15/ExpandDims:output:0*0
_output_shapes
:??????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_15/MaxPool?
max_pooling1d_15/SqueezeSqueeze!max_pooling1d_15/MaxPool:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
max_pooling1d_15/Squeeze?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDims!max_pooling1d_15/Squeeze:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_36/BiasAdd|
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_36/Relu?
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim?
max_pooling1d_16/ExpandDims
ExpandDimsconv1d_36/Relu:activations:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_16/ExpandDims?
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool?
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_16/Squeeze?
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_20/moments/mean/reduction_indices?
#batch_normalization_20/moments/meanMean!max_pooling1d_16/Squeeze:output:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_20/moments/mean?
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_20/moments/StopGradient?
0batch_normalization_20/moments/SquaredDifferenceSquaredDifference!max_pooling1d_16/Squeeze:output:04batch_normalization_20/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_20/moments/SquaredDifference?
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_20/moments/variance/reduction_indices?
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_20/moments/variance?
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_20/moments/Squeeze?
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1?
,batch_normalization_20/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49602967*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_20/AssignMovingAvg/decay?
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_20_assignmovingavg_49602967*
_output_shapes	
:?*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOp?
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49602967*
_output_shapes	
:?2,
*batch_normalization_20/AssignMovingAvg/sub?
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:05batch_normalization_20/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49602967*
_output_shapes	
:?2,
*batch_normalization_20/AssignMovingAvg/mul?
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_20_assignmovingavg_49602967.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49602967*
_output_shapes
 *
dtype02<
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_20/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49602973*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_20/AssignMovingAvg_1/decay?
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_20_assignmovingavg_1_49602973*
_output_shapes	
:?*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49602973*
_output_shapes	
:?2.
,batch_normalization_20/AssignMovingAvg_1/sub?
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:07batch_normalization_20/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49602973*
_output_shapes	
:?2.
,batch_normalization_20/AssignMovingAvg_1/mul?
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_20_assignmovingavg_1_496029730batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49602973*
_output_shapes
 *
dtype02>
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_20/batchnorm/add/y?
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/add?
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/Rsqrt?
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOp?
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/mul?
&batch_normalization_20/batchnorm/mul_1Mul!max_pooling1d_16/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/mul_1?
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/mul_2?
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp?
$batch_normalization_20/batchnorm/subSub7batch_normalization_20/batchnorm/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/sub?
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/add_1?
activation_20/ReluRelu*batch_normalization_20/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_20/Relu?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDims activation_20/Relu:activations:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_37/BiasAdd?
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_37/conv1d_37/kernel/Regularizer/SquareSquareDconv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_37/conv1d_37/kernel/Regularizer/Square?
,conv1d_37/conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_37/conv1d_37/kernel/Regularizer/Const?
*conv1d_37/conv1d_37/kernel/Regularizer/SumSum1conv1d_37/conv1d_37/kernel/Regularizer/Square:y:05conv1d_37/conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/Sum?
,conv1d_37/conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_37/conv1d_37/kernel/Regularizer/mul/x?
*conv1d_37/conv1d_37/kernel/Regularizer/mulMul5conv1d_37/conv1d_37/kernel/Regularizer/mul/x:output:03conv1d_37/conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/mul?
,conv1d_37/conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_37/conv1d_37/kernel/Regularizer/add/x?
*conv1d_37/conv1d_37/kernel/Regularizer/addAddV25conv1d_37/conv1d_37/kernel/Regularizer/add/x:output:0.conv1d_37/conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/add?
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_21/moments/mean/reduction_indices?
#batch_normalization_21/moments/meanMeanconv1d_37/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_21/moments/mean?
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_21/moments/StopGradient?
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferenceconv1d_37/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_21/moments/SquaredDifference?
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_21/moments/variance/reduction_indices?
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_21/moments/variance?
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_21/moments/Squeeze?
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_21/moments/Squeeze_1?
,batch_normalization_21/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49603019*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_21/AssignMovingAvg/decay?
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_21_assignmovingavg_49603019*
_output_shapes	
:?*
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOp?
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49603019*
_output_shapes	
:?2,
*batch_normalization_21/AssignMovingAvg/sub?
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49603019*
_output_shapes	
:?2,
*batch_normalization_21/AssignMovingAvg/mul?
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_21_assignmovingavg_49603019.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49603019*
_output_shapes
 *
dtype02<
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_21/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49603025*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_21/AssignMovingAvg_1/decay?
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_21_assignmovingavg_1_49603025*
_output_shapes	
:?*
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49603025*
_output_shapes	
:?2.
,batch_normalization_21/AssignMovingAvg_1/sub?
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49603025*
_output_shapes	
:?2.
,batch_normalization_21/AssignMovingAvg_1/mul?
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_21_assignmovingavg_1_496030250batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49603025*
_output_shapes
 *
dtype02>
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_21/batchnorm/add/y?
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/add?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/Rsqrt?
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOp?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/mul?
&batch_normalization_21/batchnorm/mul_1Mulconv1d_37/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/mul_1?
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/mul_2?
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOp?
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/sub?
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/add_1?
activation_21/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_21/Relu?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDims activation_21/Relu:activations:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_38/BiasAdd?
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_38/conv1d_38/kernel/Regularizer/SquareSquareDconv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_38/conv1d_38/kernel/Regularizer/Square?
,conv1d_38/conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_38/conv1d_38/kernel/Regularizer/Const?
*conv1d_38/conv1d_38/kernel/Regularizer/SumSum1conv1d_38/conv1d_38/kernel/Regularizer/Square:y:05conv1d_38/conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/Sum?
,conv1d_38/conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_38/conv1d_38/kernel/Regularizer/mul/x?
*conv1d_38/conv1d_38/kernel/Regularizer/mulMul5conv1d_38/conv1d_38/kernel/Regularizer/mul/x:output:03conv1d_38/conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/mul?
,conv1d_38/conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_38/conv1d_38/kernel/Regularizer/add/x?
*conv1d_38/conv1d_38/kernel/Regularizer/addAddV25conv1d_38/conv1d_38/kernel/Regularizer/add/x:output:0.conv1d_38/conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/add?

add_10/addAddV2conv1d_38/BiasAdd:output:0!max_pooling1d_16/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_10/add?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsadd_10/add:z:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_39/BiasAdd|
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_39/Relu?
max_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_17/ExpandDims/dim?
max_pooling1d_17/ExpandDims
ExpandDimsconv1d_39/Relu:activations:0(max_pooling1d_17/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_17/ExpandDims?
max_pooling1d_17/MaxPoolMaxPool$max_pooling1d_17/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_17/MaxPool?
max_pooling1d_17/SqueezeSqueeze!max_pooling1d_17/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_17/Squeeze?
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_22/moments/mean/reduction_indices?
#batch_normalization_22/moments/meanMean!max_pooling1d_17/Squeeze:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_22/moments/mean?
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_22/moments/StopGradient?
0batch_normalization_22/moments/SquaredDifferenceSquaredDifference!max_pooling1d_17/Squeeze:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_22/moments/SquaredDifference?
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_22/moments/variance/reduction_indices?
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_22/moments/variance?
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_22/moments/Squeeze?
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_22/moments/Squeeze_1?
,batch_normalization_22/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49603088*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_22/AssignMovingAvg/decay?
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_22_assignmovingavg_49603088*
_output_shapes	
:?*
dtype027
5batch_normalization_22/AssignMovingAvg/ReadVariableOp?
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49603088*
_output_shapes	
:?2,
*batch_normalization_22/AssignMovingAvg/sub?
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49603088*
_output_shapes	
:?2,
*batch_normalization_22/AssignMovingAvg/mul?
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_22_assignmovingavg_49603088.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49603088*
_output_shapes
 *
dtype02<
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_22/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49603094*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_22/AssignMovingAvg_1/decay?
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_22_assignmovingavg_1_49603094*
_output_shapes	
:?*
dtype029
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49603094*
_output_shapes	
:?2.
,batch_normalization_22/AssignMovingAvg_1/sub?
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49603094*
_output_shapes	
:?2.
,batch_normalization_22/AssignMovingAvg_1/mul?
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_22_assignmovingavg_1_496030940batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49603094*
_output_shapes
 *
dtype02>
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_22/batchnorm/add/y?
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/add?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/Rsqrt?
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_22/batchnorm/mul/ReadVariableOp?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/mul?
&batch_normalization_22/batchnorm/mul_1Mul!max_pooling1d_17/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/mul_1?
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/mul_2?
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_22/batchnorm/ReadVariableOp?
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/sub?
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/add_1?
activation_22/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_22/Relu?
6conv1d_40/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?28
6conv1d_40/required_space_to_batch_paddings/input_shape?
8conv1d_40/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8conv1d_40/required_space_to_batch_paddings/base_paddings?
3conv1d_40/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        25
3conv1d_40/required_space_to_batch_paddings/paddings?
0conv1d_40/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        22
0conv1d_40/required_space_to_batch_paddings/crops?
$conv1d_40/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/SpaceToBatchND/block_shape?
!conv1d_40/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2#
!conv1d_40/SpaceToBatchND/paddings?
conv1d_40/SpaceToBatchNDSpaceToBatchND activation_22/Relu:activations:0-conv1d_40/SpaceToBatchND/block_shape:output:0*conv1d_40/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:?????????A?2
conv1d_40/SpaceToBatchND?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDims!conv1d_40/SpaceToBatchND:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????A?2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1dConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????A?*
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*,
_output_shapes
:?????????A?*
squeeze_dims
2
conv1d_40/conv1d/Squeeze?
$conv1d_40/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/BatchToSpaceND/block_shape?
conv1d_40/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2 
conv1d_40/BatchToSpaceND/crops?
conv1d_40/BatchToSpaceNDBatchToSpaceND!conv1d_40/conv1d/Squeeze:output:0-conv1d_40/BatchToSpaceND/block_shape:output:0'conv1d_40/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BatchToSpaceND?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/BatchToSpaceND:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BiasAdd?
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_40/conv1d_40/kernel/Regularizer/SquareSquareDconv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_40/conv1d_40/kernel/Regularizer/Square?
,conv1d_40/conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_40/conv1d_40/kernel/Regularizer/Const?
*conv1d_40/conv1d_40/kernel/Regularizer/SumSum1conv1d_40/conv1d_40/kernel/Regularizer/Square:y:05conv1d_40/conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/Sum?
,conv1d_40/conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_40/conv1d_40/kernel/Regularizer/mul/x?
*conv1d_40/conv1d_40/kernel/Regularizer/mulMul5conv1d_40/conv1d_40/kernel/Regularizer/mul/x:output:03conv1d_40/conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/mul?
,conv1d_40/conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_40/conv1d_40/kernel/Regularizer/add/x?
*conv1d_40/conv1d_40/kernel/Regularizer/addAddV25conv1d_40/conv1d_40/kernel/Regularizer/add/x:output:0.conv1d_40/conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/add?
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_23/moments/mean/reduction_indices?
#batch_normalization_23/moments/meanMeanconv1d_40/BiasAdd:output:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_23/moments/mean?
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_23/moments/StopGradient?
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferenceconv1d_40/BiasAdd:output:04batch_normalization_23/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_23/moments/SquaredDifference?
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_23/moments/variance/reduction_indices?
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_23/moments/variance?
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_23/moments/Squeeze?
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_23/moments/Squeeze_1?
,batch_normalization_23/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49603151*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_23/AssignMovingAvg/decay?
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_23_assignmovingavg_49603151*
_output_shapes	
:?*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOp?
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49603151*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/sub?
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49603151*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/mul?
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_23_assignmovingavg_49603151.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49603151*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_23/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49603157*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_23/AssignMovingAvg_1/decay?
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_23_assignmovingavg_1_49603157*
_output_shapes	
:?*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49603157*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/sub?
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49603157*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/mul?
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_23_assignmovingavg_1_496031570batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49603157*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Mulconv1d_40/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/mul_1?
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/add_1?
activation_23/ReluRelu*batch_normalization_23/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_23/Relu?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDims activation_23/Relu:activations:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1dConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_41/BiasAdd?
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_41/conv1d_41/kernel/Regularizer/SquareSquareDconv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_41/conv1d_41/kernel/Regularizer/Square?
,conv1d_41/conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_41/conv1d_41/kernel/Regularizer/Const?
*conv1d_41/conv1d_41/kernel/Regularizer/SumSum1conv1d_41/conv1d_41/kernel/Regularizer/Square:y:05conv1d_41/conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/Sum?
,conv1d_41/conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_41/conv1d_41/kernel/Regularizer/mul/x?
*conv1d_41/conv1d_41/kernel/Regularizer/mulMul5conv1d_41/conv1d_41/kernel/Regularizer/mul/x:output:03conv1d_41/conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/mul?
,conv1d_41/conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_41/conv1d_41/kernel/Regularizer/add/x?
*conv1d_41/conv1d_41/kernel/Regularizer/addAddV25conv1d_41/conv1d_41/kernel/Regularizer/add/x:output:0.conv1d_41/conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/add?

add_11/addAddV2conv1d_41/BiasAdd:output:0!max_pooling1d_17/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_11/add{
interoutput/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *O???2
interoutput/dropout/Const?
interoutput/dropout/MulMuladd_11/add:z:0"interoutput/dropout/Const:output:0*
T0*-
_output_shapes
:???????????2
interoutput/dropout/Mult
interoutput/dropout/ShapeShapeadd_11/add:z:0*
T0*
_output_shapes
:2
interoutput/dropout/Shape?
0interoutput/dropout/random_uniform/RandomUniformRandomUniform"interoutput/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype022
0interoutput/dropout/random_uniform/RandomUniform?
"interoutput/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *33?>2$
"interoutput/dropout/GreaterEqual/y?
 interoutput/dropout/GreaterEqualGreaterEqual9interoutput/dropout/random_uniform/RandomUniform:output:0+interoutput/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2"
 interoutput/dropout/GreaterEqual?
interoutput/dropout/CastCast$interoutput/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
interoutput/dropout/Cast?
interoutput/dropout/Mul_1Mulinteroutput/dropout/Mul:z:0interoutput/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
interoutput/dropout/Mul_1[
reshape_16/ShapeShapeinput_6*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice?
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinput_6!reshape_16/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
reshape_16/Reshape?
"tf_op_layer_concat_5/concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_5/concat_5/axis?
tf_op_layer_concat_5/concat_5ConcatV2interoutput/dropout/Mul_1:z:0reshape_16/Reshape:output:0+tf_op_layer_concat_5/concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2
tf_op_layer_concat_5/concat_5?
(second_dropout1/Tensordot/ReadVariableOpReadVariableOp1second_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(second_dropout1/Tensordot/ReadVariableOp?
second_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout1/Tensordot/axes?
second_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout1/Tensordot/free?
second_dropout1/Tensordot/ShapeShape&tf_op_layer_concat_5/concat_5:output:0*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/Shape?
'second_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/GatherV2/axis?
"second_dropout1/Tensordot/GatherV2GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/free:output:00second_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout1/Tensordot/GatherV2?
)second_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout1/Tensordot/GatherV2_1/axis?
$second_dropout1/Tensordot/GatherV2_1GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/axes:output:02second_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout1/Tensordot/GatherV2_1?
second_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout1/Tensordot/Const?
second_dropout1/Tensordot/ProdProd+second_dropout1/Tensordot/GatherV2:output:0(second_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout1/Tensordot/Prod?
!second_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout1/Tensordot/Const_1?
 second_dropout1/Tensordot/Prod_1Prod-second_dropout1/Tensordot/GatherV2_1:output:0*second_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout1/Tensordot/Prod_1?
%second_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout1/Tensordot/concat/axis?
 second_dropout1/Tensordot/concatConcatV2'second_dropout1/Tensordot/free:output:0'second_dropout1/Tensordot/axes:output:0.second_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout1/Tensordot/concat?
second_dropout1/Tensordot/stackPack'second_dropout1/Tensordot/Prod:output:0)second_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/stack?
#second_dropout1/Tensordot/transpose	Transpose&tf_op_layer_concat_5/concat_5:output:0)second_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2%
#second_dropout1/Tensordot/transpose?
!second_dropout1/Tensordot/ReshapeReshape'second_dropout1/Tensordot/transpose:y:0(second_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout1/Tensordot/Reshape?
 second_dropout1/Tensordot/MatMulMatMul*second_dropout1/Tensordot/Reshape:output:00second_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 second_dropout1/Tensordot/MatMul?
!second_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2#
!second_dropout1/Tensordot/Const_2?
'second_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/concat_1/axis?
"second_dropout1/Tensordot/concat_1ConcatV2+second_dropout1/Tensordot/GatherV2:output:0*second_dropout1/Tensordot/Const_2:output:00second_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout1/Tensordot/concat_1?
second_dropout1/TensordotReshape*second_dropout1/Tensordot/MatMul:product:0+second_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Tensordot?
&second_dropout1/BiasAdd/ReadVariableOpReadVariableOp/second_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&second_dropout1/BiasAdd/ReadVariableOp?
second_dropout1/BiasAddBiasAdd"second_dropout1/Tensordot:output:0.second_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/BiasAdd?
second_dropout1/SigmoidSigmoid second_dropout1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Sigmoid?
%one_dropout1/Tensordot/ReadVariableOpReadVariableOp.one_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%one_dropout1/Tensordot/ReadVariableOp?
one_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout1/Tensordot/axes?
one_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout1/Tensordot/free?
one_dropout1/Tensordot/ShapeShapeinteroutput/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
one_dropout1/Tensordot/Shape?
$one_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/GatherV2/axis?
one_dropout1/Tensordot/GatherV2GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/free:output:0-one_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout1/Tensordot/GatherV2?
&one_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout1/Tensordot/GatherV2_1/axis?
!one_dropout1/Tensordot/GatherV2_1GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/axes:output:0/one_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout1/Tensordot/GatherV2_1?
one_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout1/Tensordot/Const?
one_dropout1/Tensordot/ProdProd(one_dropout1/Tensordot/GatherV2:output:0%one_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod?
one_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout1/Tensordot/Const_1?
one_dropout1/Tensordot/Prod_1Prod*one_dropout1/Tensordot/GatherV2_1:output:0'one_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod_1?
"one_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout1/Tensordot/concat/axis?
one_dropout1/Tensordot/concatConcatV2$one_dropout1/Tensordot/free:output:0$one_dropout1/Tensordot/axes:output:0+one_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/concat?
one_dropout1/Tensordot/stackPack$one_dropout1/Tensordot/Prod:output:0&one_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/stack?
 one_dropout1/Tensordot/transpose	Transposeinteroutput/dropout/Mul_1:z:0&one_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2"
 one_dropout1/Tensordot/transpose?
one_dropout1/Tensordot/ReshapeReshape$one_dropout1/Tensordot/transpose:y:0%one_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout1/Tensordot/Reshape?
one_dropout1/Tensordot/MatMulMatMul'one_dropout1/Tensordot/Reshape:output:0-one_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
one_dropout1/Tensordot/MatMul?
one_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2 
one_dropout1/Tensordot/Const_2?
$one_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/concat_1/axis?
one_dropout1/Tensordot/concat_1ConcatV2(one_dropout1/Tensordot/GatherV2:output:0'one_dropout1/Tensordot/Const_2:output:0-one_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout1/Tensordot/concat_1?
one_dropout1/TensordotReshape'one_dropout1/Tensordot/MatMul:product:0(one_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Tensordot?
#one_dropout1/BiasAdd/ReadVariableOpReadVariableOp,one_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#one_dropout1/BiasAdd/ReadVariableOp?
one_dropout1/BiasAddBiasAddone_dropout1/Tensordot:output:0+one_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/BiasAdd?
"one_dropout1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Max/reduction_indices?
one_dropout1/MaxMaxone_dropout1/BiasAdd:output:0+one_dropout1/Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Max?
one_dropout1/subSubone_dropout1/BiasAdd:output:0one_dropout1/Max:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/subx
one_dropout1/ExpExpone_dropout1/sub:z:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Exp?
"one_dropout1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Sum/reduction_indices?
one_dropout1/SumSumone_dropout1/Exp:y:0+one_dropout1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Sum?
one_dropout1/truedivRealDivone_dropout1/Exp:y:0one_dropout1/Sum:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/truedivo
reshape_17/ShapeShapesecond_dropout1/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slice?
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_17/Reshape/shape/1{
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapesecond_dropout1/Sigmoid:y:0!reshape_17/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_17/Reshapel
reshape_15/ShapeShapeone_dropout1/truediv:z:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slice?
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_15/Reshape/shape/1{
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_15/Reshape/shape/2?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeone_dropout1/truediv:z:0!reshape_15/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_15/Reshape?
(second_dropout2/Tensordot/ReadVariableOpReadVariableOp1second_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02*
(second_dropout2/Tensordot/ReadVariableOp?
second_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout2/Tensordot/axes?
second_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout2/Tensordot/free?
second_dropout2/Tensordot/ShapeShapereshape_17/Reshape:output:0*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/Shape?
'second_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/GatherV2/axis?
"second_dropout2/Tensordot/GatherV2GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/free:output:00second_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout2/Tensordot/GatherV2?
)second_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout2/Tensordot/GatherV2_1/axis?
$second_dropout2/Tensordot/GatherV2_1GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/axes:output:02second_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout2/Tensordot/GatherV2_1?
second_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout2/Tensordot/Const?
second_dropout2/Tensordot/ProdProd+second_dropout2/Tensordot/GatherV2:output:0(second_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout2/Tensordot/Prod?
!second_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout2/Tensordot/Const_1?
 second_dropout2/Tensordot/Prod_1Prod-second_dropout2/Tensordot/GatherV2_1:output:0*second_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout2/Tensordot/Prod_1?
%second_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout2/Tensordot/concat/axis?
 second_dropout2/Tensordot/concatConcatV2'second_dropout2/Tensordot/free:output:0'second_dropout2/Tensordot/axes:output:0.second_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout2/Tensordot/concat?
second_dropout2/Tensordot/stackPack'second_dropout2/Tensordot/Prod:output:0)second_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/stack?
#second_dropout2/Tensordot/transpose	Transposereshape_17/Reshape:output:0)second_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2%
#second_dropout2/Tensordot/transpose?
!second_dropout2/Tensordot/ReshapeReshape'second_dropout2/Tensordot/transpose:y:0(second_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout2/Tensordot/Reshape?
 second_dropout2/Tensordot/MatMulMatMul*second_dropout2/Tensordot/Reshape:output:00second_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 second_dropout2/Tensordot/MatMul?
!second_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!second_dropout2/Tensordot/Const_2?
'second_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/concat_1/axis?
"second_dropout2/Tensordot/concat_1ConcatV2+second_dropout2/Tensordot/GatherV2:output:0*second_dropout2/Tensordot/Const_2:output:00second_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout2/Tensordot/concat_1?
second_dropout2/TensordotReshape*second_dropout2/Tensordot/MatMul:product:0+second_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Tensordot?
&second_dropout2/BiasAdd/ReadVariableOpReadVariableOp/second_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&second_dropout2/BiasAdd/ReadVariableOp?
second_dropout2/BiasAddBiasAdd"second_dropout2/Tensordot:output:0.second_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/BiasAdd?
second_dropout2/SigmoidSigmoid second_dropout2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Sigmoid?
%one_dropout2/Tensordot/ReadVariableOpReadVariableOp.one_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02'
%one_dropout2/Tensordot/ReadVariableOp?
one_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout2/Tensordot/axes?
one_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout2/Tensordot/free?
one_dropout2/Tensordot/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
one_dropout2/Tensordot/Shape?
$one_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/GatherV2/axis?
one_dropout2/Tensordot/GatherV2GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/free:output:0-one_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout2/Tensordot/GatherV2?
&one_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout2/Tensordot/GatherV2_1/axis?
!one_dropout2/Tensordot/GatherV2_1GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/axes:output:0/one_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout2/Tensordot/GatherV2_1?
one_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout2/Tensordot/Const?
one_dropout2/Tensordot/ProdProd(one_dropout2/Tensordot/GatherV2:output:0%one_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod?
one_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout2/Tensordot/Const_1?
one_dropout2/Tensordot/Prod_1Prod*one_dropout2/Tensordot/GatherV2_1:output:0'one_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod_1?
"one_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout2/Tensordot/concat/axis?
one_dropout2/Tensordot/concatConcatV2$one_dropout2/Tensordot/free:output:0$one_dropout2/Tensordot/axes:output:0+one_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/concat?
one_dropout2/Tensordot/stackPack$one_dropout2/Tensordot/Prod:output:0&one_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/stack?
 one_dropout2/Tensordot/transpose	Transposereshape_15/Reshape:output:0&one_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2"
 one_dropout2/Tensordot/transpose?
one_dropout2/Tensordot/ReshapeReshape$one_dropout2/Tensordot/transpose:y:0%one_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout2/Tensordot/Reshape?
one_dropout2/Tensordot/MatMulMatMul'one_dropout2/Tensordot/Reshape:output:0-one_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
one_dropout2/Tensordot/MatMul?
one_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2 
one_dropout2/Tensordot/Const_2?
$one_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/concat_1/axis?
one_dropout2/Tensordot/concat_1ConcatV2(one_dropout2/Tensordot/GatherV2:output:0'one_dropout2/Tensordot/Const_2:output:0-one_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout2/Tensordot/concat_1?
one_dropout2/TensordotReshape'one_dropout2/Tensordot/MatMul:product:0(one_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Tensordot?
#one_dropout2/BiasAdd/ReadVariableOpReadVariableOp,one_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#one_dropout2/BiasAdd/ReadVariableOp?
one_dropout2/BiasAddBiasAddone_dropout2/Tensordot:output:0+one_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/BiasAdd?
"one_dropout2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Max/reduction_indices?
one_dropout2/MaxMaxone_dropout2/BiasAdd:output:0+one_dropout2/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Max?
one_dropout2/subSubone_dropout2/BiasAdd:output:0one_dropout2/Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/sub?
one_dropout2/ExpExpone_dropout2/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Exp?
"one_dropout2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Sum/reduction_indices?
one_dropout2/SumSumone_dropout2/Exp:y:0+one_dropout2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Sum?
one_dropout2/truedivRealDivone_dropout2/Exp:y:0one_dropout2/Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/truedivu
second_output/ShapeShapesecond_dropout2/Sigmoid:y:0*
T0*
_output_shapes
:2
second_output/Shape?
!second_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!second_output/strided_slice/stack?
#second_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_1?
#second_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_2?
second_output/strided_sliceStridedSlicesecond_output/Shape:output:0*second_output/strided_slice/stack:output:0,second_output/strided_slice/stack_1:output:0,second_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
second_output/strided_slice?
second_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
second_output/Reshape/shape/1?
second_output/Reshape/shapePack$second_output/strided_slice:output:0&second_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
second_output/Reshape/shape?
second_output/ReshapeReshapesecond_dropout2/Sigmoid:y:0$second_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
second_output/Reshapel
one_output/ShapeShapeone_dropout2/truediv:z:0*
T0*
_output_shapes
:2
one_output/Shape?
one_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
one_output/strided_slice/stack?
 one_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_1?
 one_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_2?
one_output/strided_sliceStridedSliceone_output/Shape:output:0'one_output/strided_slice/stack:output:0)one_output/strided_slice/stack_1:output:0)one_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
one_output/strided_slice?
one_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
one_output/Reshape/shape/1?
one_output/Reshape/shapePack!one_output/strided_slice:output:0#one_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
one_output/Reshape/shape?
one_output/ReshapeReshapeone_dropout2/truediv:z:0!one_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
one_output/Reshape?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/add?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/add?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/add?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/add?
IdentityIdentityone_output/Reshape:output:0;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:??????????????????2

Identity?

Identity_1Identitysecond_output/Reshape:output:0;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????:::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
?+
?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604154

inputs
assignmovingavg_49604129
assignmovingavg_1_49604135)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604129*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604129*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604129*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604129*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604129AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604129*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604135*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604135*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604135*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604135*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604135AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604135*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
p
D__inference_add_11_layer_call_and_return_conditional_losses_49605212
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:???????????2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:???????????:???????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?(
?
G__inference_conv1d_40_layer_call_and_return_conditional_losses_49600846

inputs0
,required_space_to_batch_paddings_block_shape/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??
,required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?2.
,required_space_to_batch_paddings/input_shape?
.required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        20
.required_space_to_batch_paddings/base_paddings?
)required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2+
)required_space_to_batch_paddings/paddings?
&required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2(
&required_space_to_batch_paddings/crops?
SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
SpaceToBatchND/block_shape?
SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2
SpaceToBatchND/paddings?
SpaceToBatchNDSpaceToBatchNDinputs#SpaceToBatchND/block_shape:output:0 SpaceToBatchND/paddings:output:0*
T0*5
_output_shapes#
!:???????????????????2
SpaceToBatchNDp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsSpaceToBatchND:output:0conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
BatchToSpaceND/block_shape?
BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2
BatchToSpaceND/crops?
BatchToSpaceNDBatchToSpaceNDconv1d/Squeeze:output:0#BatchToSpaceND/block_shape:output:0BatchToSpaceND/crops:output:0*
T0*5
_output_shapes#
!:???????????????????2
BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddBatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:???????????????????::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_reshape_16_layer_call_and_return_conditional_losses_49605265

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape|
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2	
Reshapeq
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

I
-__inference_one_output_layer_call_fn_49605640

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
,__inference_conv1d_41_layer_call_fn_49601043

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
U
)__inference_add_11_layer_call_fn_49605218
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:???????????2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:???????????:???????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
~
R__inference_tf_op_layer_concat_5_layer_call_and_return_conditional_losses_49605285
inputs_0
inputs_1
identityi
concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_5/axis?
concat_5ConcatV2inputs_0inputs_1concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2

concat_5k
IdentityIdentityconcat_5:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:???????????:??????????????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/1
?$
?
/__inference_one_dropout1_layer_call_fn_49605366

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
Maxh
subSubBiasAdd:output:0Max:output:0*
T0*,
_output_shapes
:??????????@2
subQ
ExpExpsub:z:0*
T0*,
_output_shapes
:??????????@2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
Sumk
truedivRealDivExp:y:0Sum:output:0*
T0*,
_output_shapes
:??????????@2	
truedivd
IdentityIdentitytruediv:z:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????:::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_activation_20_layer_call_fn_49604352

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604854

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?%
?
J__inference_one_dropout2_layer_call_and_return_conditional_losses_49605517

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Maxp
subSubBiasAdd:output:0Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
subY
ExpExpsub:z:0*
T0*4
_output_shapes"
 :??????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sums
truedivRealDivExp:y:0Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2	
truedivl
IdentityIdentitytruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????A:::] Y
5
_output_shapes#
!:???????????????????A
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_reshape_17_layer_call_fn_49605480

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape}
ReshapeReshapeinputsReshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:???????????????????A2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_20_layer_call_fn_49604230

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
g
I__inference_interoutput_layer_call_and_return_conditional_losses_49605235

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_conv1d_41_layer_call_and_return_conditional_losses_49601019

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_conv1d_38_layer_call_and_return_conditional_losses_49600621

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
h
I__inference_interoutput_layer_call_and_return_conditional_losses_49605230

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *O???2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *33?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
M__inference_second_dropout1_layer_call_and_return_conditional_losses_49605397

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAddf
SigmoidSigmoidBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2	
Sigmoidd
IdentityIdentitySigmoid:y:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????:::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?%
?
/__inference_one_dropout2_layer_call_fn_49605554

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Maxp
subSubBiasAdd:output:0Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
subY
ExpExpsub:z:0*
T0*4
_output_shapes"
 :??????????????????2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
Sums
truedivRealDivExp:y:0Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2	
truedivl
IdentityIdentitytruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????A:::] Y
5
_output_shapes#
!:???????????????????A
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
g
K__inference_second_output_layer_call_and_return_conditional_losses_49605652

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv1d_36_layer_call_and_return_conditional_losses_49600286

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@:::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
#__inference__wrapped_model_49600217
input_6A
=model_5_conv1d_35_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_35_biasadd_readvariableop_resourceA
=model_5_conv1d_36_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_36_biasadd_readvariableop_resourceD
@model_5_batch_normalization_20_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_20_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_20_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_20_batchnorm_readvariableop_2_resourceA
=model_5_conv1d_37_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_37_biasadd_readvariableop_resourceD
@model_5_batch_normalization_21_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_21_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_21_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_21_batchnorm_readvariableop_2_resourceA
=model_5_conv1d_38_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_38_biasadd_readvariableop_resourceA
=model_5_conv1d_39_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_39_biasadd_readvariableop_resourceD
@model_5_batch_normalization_22_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_22_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_22_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_22_batchnorm_readvariableop_2_resourceB
>model_5_conv1d_40_required_space_to_batch_paddings_block_shapeA
=model_5_conv1d_40_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_40_biasadd_readvariableop_resourceD
@model_5_batch_normalization_23_batchnorm_readvariableop_resourceH
Dmodel_5_batch_normalization_23_batchnorm_mul_readvariableop_resourceF
Bmodel_5_batch_normalization_23_batchnorm_readvariableop_1_resourceF
Bmodel_5_batch_normalization_23_batchnorm_readvariableop_2_resourceA
=model_5_conv1d_41_conv1d_expanddims_1_readvariableop_resource5
1model_5_conv1d_41_biasadd_readvariableop_resource=
9model_5_second_dropout1_tensordot_readvariableop_resource;
7model_5_second_dropout1_biasadd_readvariableop_resource:
6model_5_one_dropout1_tensordot_readvariableop_resource8
4model_5_one_dropout1_biasadd_readvariableop_resource=
9model_5_second_dropout2_tensordot_readvariableop_resource;
7model_5_second_dropout2_biasadd_readvariableop_resource:
6model_5_one_dropout2_tensordot_readvariableop_resource8
4model_5_one_dropout2_biasadd_readvariableop_resource
identity

identity_1??
'model_5/conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_35/conv1d/ExpandDims/dim?
#model_5/conv1d_35/conv1d/ExpandDims
ExpandDimsinput_60model_5/conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2%
#model_5/conv1d_35/conv1d/ExpandDims?
4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype026
4model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_35/conv1d/ExpandDims_1/dim?
%model_5/conv1d_35/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2'
%model_5/conv1d_35/conv1d/ExpandDims_1?
model_5/conv1d_35/conv1dConv2D,model_5/conv1d_35/conv1d/ExpandDims:output:0.model_5/conv1d_35/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
model_5/conv1d_35/conv1d?
 model_5/conv1d_35/conv1d/SqueezeSqueeze!model_5/conv1d_35/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2"
 model_5/conv1d_35/conv1d/Squeeze?
(model_5/conv1d_35/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_5/conv1d_35/BiasAdd/ReadVariableOp?
model_5/conv1d_35/BiasAddBiasAdd)model_5/conv1d_35/conv1d/Squeeze:output:00model_5/conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
model_5/conv1d_35/BiasAdd?
model_5/conv1d_35/ReluRelu"model_5/conv1d_35/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
model_5/conv1d_35/Relu?
'model_5/max_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/max_pooling1d_15/ExpandDims/dim?
#model_5/max_pooling1d_15/ExpandDims
ExpandDims$model_5/conv1d_35/Relu:activations:00model_5/max_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2%
#model_5/max_pooling1d_15/ExpandDims?
 model_5/max_pooling1d_15/MaxPoolMaxPool,model_5/max_pooling1d_15/ExpandDims:output:0*0
_output_shapes
:??????????@*
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling1d_15/MaxPool?
 model_5/max_pooling1d_15/SqueezeSqueeze)model_5/max_pooling1d_15/MaxPool:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2"
 model_5/max_pooling1d_15/Squeeze?
'model_5/conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_36/conv1d/ExpandDims/dim?
#model_5/conv1d_36/conv1d/ExpandDims
ExpandDims)model_5/max_pooling1d_15/Squeeze:output:00model_5/conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2%
#model_5/conv1d_36/conv1d/ExpandDims?
4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_36_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype026
4model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_36/conv1d/ExpandDims_1/dim?
%model_5/conv1d_36/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2'
%model_5/conv1d_36/conv1d/ExpandDims_1?
model_5/conv1d_36/conv1dConv2D,model_5/conv1d_36/conv1d/ExpandDims:output:0.model_5/conv1d_36/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_5/conv1d_36/conv1d?
 model_5/conv1d_36/conv1d/SqueezeSqueeze!model_5/conv1d_36/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/conv1d_36/conv1d/Squeeze?
(model_5/conv1d_36/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_36/BiasAdd/ReadVariableOp?
model_5/conv1d_36/BiasAddBiasAdd)model_5/conv1d_36/conv1d/Squeeze:output:00model_5/conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_36/BiasAdd?
model_5/conv1d_36/ReluRelu"model_5/conv1d_36/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_36/Relu?
'model_5/max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/max_pooling1d_16/ExpandDims/dim?
#model_5/max_pooling1d_16/ExpandDims
ExpandDims$model_5/conv1d_36/Relu:activations:00model_5/max_pooling1d_16/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/max_pooling1d_16/ExpandDims?
 model_5/max_pooling1d_16/MaxPoolMaxPool,model_5/max_pooling1d_16/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling1d_16/MaxPool?
 model_5/max_pooling1d_16/SqueezeSqueeze)model_5/max_pooling1d_16/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/max_pooling1d_16/Squeeze?
7model_5/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_5/batch_normalization_20/batchnorm/ReadVariableOp?
.model_5/batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_5/batch_normalization_20/batchnorm/add/y?
,model_5/batch_normalization_20/batchnorm/addAddV2?model_5/batch_normalization_20/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_20/batchnorm/add?
.model_5/batch_normalization_20/batchnorm/RsqrtRsqrt0model_5/batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_20/batchnorm/Rsqrt?
;model_5/batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_5/batch_normalization_20/batchnorm/mul/ReadVariableOp?
,model_5/batch_normalization_20/batchnorm/mulMul2model_5/batch_normalization_20/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_20/batchnorm/mul?
.model_5/batch_normalization_20/batchnorm/mul_1Mul)model_5/max_pooling1d_16/Squeeze:output:00model_5/batch_normalization_20/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_20/batchnorm/mul_1?
9model_5/batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_20/batchnorm/ReadVariableOp_1?
.model_5/batch_normalization_20/batchnorm/mul_2MulAmodel_5/batch_normalization_20/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_20/batchnorm/mul_2?
9model_5/batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_20/batchnorm/ReadVariableOp_2?
,model_5/batch_normalization_20/batchnorm/subSubAmodel_5/batch_normalization_20/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_20/batchnorm/sub?
.model_5/batch_normalization_20/batchnorm/add_1AddV22model_5/batch_normalization_20/batchnorm/mul_1:z:00model_5/batch_normalization_20/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_20/batchnorm/add_1?
model_5/activation_20/ReluRelu2model_5/batch_normalization_20/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
model_5/activation_20/Relu?
'model_5/conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_37/conv1d/ExpandDims/dim?
#model_5/conv1d_37/conv1d/ExpandDims
ExpandDims(model_5/activation_20/Relu:activations:00model_5/conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/conv1d_37/conv1d/ExpandDims?
4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype026
4model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_37/conv1d/ExpandDims_1/dim?
%model_5/conv1d_37/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2'
%model_5/conv1d_37/conv1d/ExpandDims_1?
model_5/conv1d_37/conv1dConv2D,model_5/conv1d_37/conv1d/ExpandDims:output:0.model_5/conv1d_37/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model_5/conv1d_37/conv1d?
 model_5/conv1d_37/conv1d/SqueezeSqueeze!model_5/conv1d_37/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/conv1d_37/conv1d/Squeeze?
(model_5/conv1d_37/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_37/BiasAdd/ReadVariableOp?
model_5/conv1d_37/BiasAddBiasAdd)model_5/conv1d_37/conv1d/Squeeze:output:00model_5/conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_37/BiasAdd?
7model_5/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_5/batch_normalization_21/batchnorm/ReadVariableOp?
.model_5/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_5/batch_normalization_21/batchnorm/add/y?
,model_5/batch_normalization_21/batchnorm/addAddV2?model_5/batch_normalization_21/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_21/batchnorm/add?
.model_5/batch_normalization_21/batchnorm/RsqrtRsqrt0model_5/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_21/batchnorm/Rsqrt?
;model_5/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_5/batch_normalization_21/batchnorm/mul/ReadVariableOp?
,model_5/batch_normalization_21/batchnorm/mulMul2model_5/batch_normalization_21/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_21/batchnorm/mul?
.model_5/batch_normalization_21/batchnorm/mul_1Mul"model_5/conv1d_37/BiasAdd:output:00model_5/batch_normalization_21/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_21/batchnorm/mul_1?
9model_5/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_21/batchnorm/ReadVariableOp_1?
.model_5/batch_normalization_21/batchnorm/mul_2MulAmodel_5/batch_normalization_21/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_21/batchnorm/mul_2?
9model_5/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_21/batchnorm/ReadVariableOp_2?
,model_5/batch_normalization_21/batchnorm/subSubAmodel_5/batch_normalization_21/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_21/batchnorm/sub?
.model_5/batch_normalization_21/batchnorm/add_1AddV22model_5/batch_normalization_21/batchnorm/mul_1:z:00model_5/batch_normalization_21/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_21/batchnorm/add_1?
model_5/activation_21/ReluRelu2model_5/batch_normalization_21/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
model_5/activation_21/Relu?
'model_5/conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_38/conv1d/ExpandDims/dim?
#model_5/conv1d_38/conv1d/ExpandDims
ExpandDims(model_5/activation_21/Relu:activations:00model_5/conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/conv1d_38/conv1d/ExpandDims?
4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype026
4model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_38/conv1d/ExpandDims_1/dim?
%model_5/conv1d_38/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2'
%model_5/conv1d_38/conv1d/ExpandDims_1?
model_5/conv1d_38/conv1dConv2D,model_5/conv1d_38/conv1d/ExpandDims:output:0.model_5/conv1d_38/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_5/conv1d_38/conv1d?
 model_5/conv1d_38/conv1d/SqueezeSqueeze!model_5/conv1d_38/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/conv1d_38/conv1d/Squeeze?
(model_5/conv1d_38/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_38/BiasAdd/ReadVariableOp?
model_5/conv1d_38/BiasAddBiasAdd)model_5/conv1d_38/conv1d/Squeeze:output:00model_5/conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_38/BiasAdd?
model_5/add_10/addAddV2"model_5/conv1d_38/BiasAdd:output:0)model_5/max_pooling1d_16/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
model_5/add_10/add?
'model_5/conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_39/conv1d/ExpandDims/dim?
#model_5/conv1d_39/conv1d/ExpandDims
ExpandDimsmodel_5/add_10/add:z:00model_5/conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/conv1d_39/conv1d/ExpandDims?
4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_39_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype026
4model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_39/conv1d/ExpandDims_1/dim?
%model_5/conv1d_39/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2'
%model_5/conv1d_39/conv1d/ExpandDims_1?
model_5/conv1d_39/conv1dConv2D,model_5/conv1d_39/conv1d/ExpandDims:output:0.model_5/conv1d_39/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_5/conv1d_39/conv1d?
 model_5/conv1d_39/conv1d/SqueezeSqueeze!model_5/conv1d_39/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/conv1d_39/conv1d/Squeeze?
(model_5/conv1d_39/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_39/BiasAdd/ReadVariableOp?
model_5/conv1d_39/BiasAddBiasAdd)model_5/conv1d_39/conv1d/Squeeze:output:00model_5/conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_39/BiasAdd?
model_5/conv1d_39/ReluRelu"model_5/conv1d_39/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_39/Relu?
'model_5/max_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/max_pooling1d_17/ExpandDims/dim?
#model_5/max_pooling1d_17/ExpandDims
ExpandDims$model_5/conv1d_39/Relu:activations:00model_5/max_pooling1d_17/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/max_pooling1d_17/ExpandDims?
 model_5/max_pooling1d_17/MaxPoolMaxPool,model_5/max_pooling1d_17/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling1d_17/MaxPool?
 model_5/max_pooling1d_17/SqueezeSqueeze)model_5/max_pooling1d_17/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/max_pooling1d_17/Squeeze?
7model_5/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_5/batch_normalization_22/batchnorm/ReadVariableOp?
.model_5/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_5/batch_normalization_22/batchnorm/add/y?
,model_5/batch_normalization_22/batchnorm/addAddV2?model_5/batch_normalization_22/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_22/batchnorm/add?
.model_5/batch_normalization_22/batchnorm/RsqrtRsqrt0model_5/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_22/batchnorm/Rsqrt?
;model_5/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_5/batch_normalization_22/batchnorm/mul/ReadVariableOp?
,model_5/batch_normalization_22/batchnorm/mulMul2model_5/batch_normalization_22/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_22/batchnorm/mul?
.model_5/batch_normalization_22/batchnorm/mul_1Mul)model_5/max_pooling1d_17/Squeeze:output:00model_5/batch_normalization_22/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_22/batchnorm/mul_1?
9model_5/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_22/batchnorm/ReadVariableOp_1?
.model_5/batch_normalization_22/batchnorm/mul_2MulAmodel_5/batch_normalization_22/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_22/batchnorm/mul_2?
9model_5/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_22/batchnorm/ReadVariableOp_2?
,model_5/batch_normalization_22/batchnorm/subSubAmodel_5/batch_normalization_22/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_22/batchnorm/sub?
.model_5/batch_normalization_22/batchnorm/add_1AddV22model_5/batch_normalization_22/batchnorm/mul_1:z:00model_5/batch_normalization_22/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_22/batchnorm/add_1?
model_5/activation_22/ReluRelu2model_5/batch_normalization_22/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
model_5/activation_22/Relu?
>model_5/conv1d_40/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?2@
>model_5/conv1d_40/required_space_to_batch_paddings/input_shape?
@model_5/conv1d_40/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2B
@model_5/conv1d_40/required_space_to_batch_paddings/base_paddings?
;model_5/conv1d_40/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2=
;model_5/conv1d_40/required_space_to_batch_paddings/paddings?
8model_5/conv1d_40/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8model_5/conv1d_40/required_space_to_batch_paddings/crops?
,model_5/conv1d_40/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2.
,model_5/conv1d_40/SpaceToBatchND/block_shape?
)model_5/conv1d_40/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2+
)model_5/conv1d_40/SpaceToBatchND/paddings?
 model_5/conv1d_40/SpaceToBatchNDSpaceToBatchND(model_5/activation_22/Relu:activations:05model_5/conv1d_40/SpaceToBatchND/block_shape:output:02model_5/conv1d_40/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:?????????A?2"
 model_5/conv1d_40/SpaceToBatchND?
'model_5/conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_40/conv1d/ExpandDims/dim?
#model_5/conv1d_40/conv1d/ExpandDims
ExpandDims)model_5/conv1d_40/SpaceToBatchND:output:00model_5/conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????A?2%
#model_5/conv1d_40/conv1d/ExpandDims?
4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype026
4model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_40/conv1d/ExpandDims_1/dim?
%model_5/conv1d_40/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2'
%model_5/conv1d_40/conv1d/ExpandDims_1?
model_5/conv1d_40/conv1dConv2D,model_5/conv1d_40/conv1d/ExpandDims:output:0.model_5/conv1d_40/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????A?*
paddingVALID*
strides
2
model_5/conv1d_40/conv1d?
 model_5/conv1d_40/conv1d/SqueezeSqueeze!model_5/conv1d_40/conv1d:output:0*
T0*,
_output_shapes
:?????????A?*
squeeze_dims
2"
 model_5/conv1d_40/conv1d/Squeeze?
,model_5/conv1d_40/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2.
,model_5/conv1d_40/BatchToSpaceND/block_shape?
&model_5/conv1d_40/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2(
&model_5/conv1d_40/BatchToSpaceND/crops?
 model_5/conv1d_40/BatchToSpaceNDBatchToSpaceND)model_5/conv1d_40/conv1d/Squeeze:output:05model_5/conv1d_40/BatchToSpaceND/block_shape:output:0/model_5/conv1d_40/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:???????????2"
 model_5/conv1d_40/BatchToSpaceND?
(model_5/conv1d_40/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_40/BiasAdd/ReadVariableOp?
model_5/conv1d_40/BiasAddBiasAdd)model_5/conv1d_40/BatchToSpaceND:output:00model_5/conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_40/BiasAdd?
7model_5/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_5/batch_normalization_23/batchnorm/ReadVariableOp?
.model_5/batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_5/batch_normalization_23/batchnorm/add/y?
,model_5/batch_normalization_23/batchnorm/addAddV2?model_5/batch_normalization_23/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_23/batchnorm/add?
.model_5/batch_normalization_23/batchnorm/RsqrtRsqrt0model_5/batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_23/batchnorm/Rsqrt?
;model_5/batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_5/batch_normalization_23/batchnorm/mul/ReadVariableOp?
,model_5/batch_normalization_23/batchnorm/mulMul2model_5/batch_normalization_23/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_23/batchnorm/mul?
.model_5/batch_normalization_23/batchnorm/mul_1Mul"model_5/conv1d_40/BiasAdd:output:00model_5/batch_normalization_23/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_23/batchnorm/mul_1?
9model_5/batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_23/batchnorm/ReadVariableOp_1?
.model_5/batch_normalization_23/batchnorm/mul_2MulAmodel_5/batch_normalization_23/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_5/batch_normalization_23/batchnorm/mul_2?
9model_5/batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_5/batch_normalization_23/batchnorm/ReadVariableOp_2?
,model_5/batch_normalization_23/batchnorm/subSubAmodel_5/batch_normalization_23/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_5/batch_normalization_23/batchnorm/sub?
.model_5/batch_normalization_23/batchnorm/add_1AddV22model_5/batch_normalization_23/batchnorm/mul_1:z:00model_5/batch_normalization_23/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????20
.model_5/batch_normalization_23/batchnorm/add_1?
model_5/activation_23/ReluRelu2model_5/batch_normalization_23/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
model_5/activation_23/Relu?
'model_5/conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_5/conv1d_41/conv1d/ExpandDims/dim?
#model_5/conv1d_41/conv1d/ExpandDims
ExpandDims(model_5/activation_23/Relu:activations:00model_5/conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2%
#model_5/conv1d_41/conv1d/ExpandDims?
4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=model_5_conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype026
4model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
)model_5/conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_5/conv1d_41/conv1d/ExpandDims_1/dim?
%model_5/conv1d_41/conv1d/ExpandDims_1
ExpandDims<model_5/conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:02model_5/conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2'
%model_5/conv1d_41/conv1d/ExpandDims_1?
model_5/conv1d_41/conv1dConv2D,model_5/conv1d_41/conv1d/ExpandDims:output:0.model_5/conv1d_41/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_5/conv1d_41/conv1d?
 model_5/conv1d_41/conv1d/SqueezeSqueeze!model_5/conv1d_41/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2"
 model_5/conv1d_41/conv1d/Squeeze?
(model_5/conv1d_41/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv1d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_5/conv1d_41/BiasAdd/ReadVariableOp?
model_5/conv1d_41/BiasAddBiasAdd)model_5/conv1d_41/conv1d/Squeeze:output:00model_5/conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_5/conv1d_41/BiasAdd?
model_5/add_11/addAddV2"model_5/conv1d_41/BiasAdd:output:0)model_5/max_pooling1d_17/Squeeze:output:0*
T0*-
_output_shapes
:???????????2
model_5/add_11/add?
model_5/interoutput/IdentityIdentitymodel_5/add_11/add:z:0*
T0*-
_output_shapes
:???????????2
model_5/interoutput/Identityk
model_5/reshape_16/ShapeShapeinput_6*
T0*
_output_shapes
:2
model_5/reshape_16/Shape?
&model_5/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/reshape_16/strided_slice/stack?
(model_5/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_16/strided_slice/stack_1?
(model_5/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_16/strided_slice/stack_2?
 model_5/reshape_16/strided_sliceStridedSlice!model_5/reshape_16/Shape:output:0/model_5/reshape_16/strided_slice/stack:output:01model_5/reshape_16/strided_slice/stack_1:output:01model_5/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_5/reshape_16/strided_slice?
"model_5/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_5/reshape_16/Reshape/shape/1?
"model_5/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_5/reshape_16/Reshape/shape/2?
 model_5/reshape_16/Reshape/shapePack)model_5/reshape_16/strided_slice:output:0+model_5/reshape_16/Reshape/shape/1:output:0+model_5/reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_5/reshape_16/Reshape/shape?
model_5/reshape_16/ReshapeReshapeinput_6)model_5/reshape_16/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_5/reshape_16/Reshape?
*model_5/tf_op_layer_concat_5/concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_5/tf_op_layer_concat_5/concat_5/axis?
%model_5/tf_op_layer_concat_5/concat_5ConcatV2%model_5/interoutput/Identity:output:0#model_5/reshape_16/Reshape:output:03model_5/tf_op_layer_concat_5/concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2'
%model_5/tf_op_layer_concat_5/concat_5?
0model_5/second_dropout1/Tensordot/ReadVariableOpReadVariableOp9model_5_second_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype022
0model_5/second_dropout1/Tensordot/ReadVariableOp?
&model_5/second_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&model_5/second_dropout1/Tensordot/axes?
&model_5/second_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_5/second_dropout1/Tensordot/free?
'model_5/second_dropout1/Tensordot/ShapeShape.model_5/tf_op_layer_concat_5/concat_5:output:0*
T0*
_output_shapes
:2)
'model_5/second_dropout1/Tensordot/Shape?
/model_5/second_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_5/second_dropout1/Tensordot/GatherV2/axis?
*model_5/second_dropout1/Tensordot/GatherV2GatherV20model_5/second_dropout1/Tensordot/Shape:output:0/model_5/second_dropout1/Tensordot/free:output:08model_5/second_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*model_5/second_dropout1/Tensordot/GatherV2?
1model_5/second_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model_5/second_dropout1/Tensordot/GatherV2_1/axis?
,model_5/second_dropout1/Tensordot/GatherV2_1GatherV20model_5/second_dropout1/Tensordot/Shape:output:0/model_5/second_dropout1/Tensordot/axes:output:0:model_5/second_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,model_5/second_dropout1/Tensordot/GatherV2_1?
'model_5/second_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_5/second_dropout1/Tensordot/Const?
&model_5/second_dropout1/Tensordot/ProdProd3model_5/second_dropout1/Tensordot/GatherV2:output:00model_5/second_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&model_5/second_dropout1/Tensordot/Prod?
)model_5/second_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)model_5/second_dropout1/Tensordot/Const_1?
(model_5/second_dropout1/Tensordot/Prod_1Prod5model_5/second_dropout1/Tensordot/GatherV2_1:output:02model_5/second_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(model_5/second_dropout1/Tensordot/Prod_1?
-model_5/second_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_5/second_dropout1/Tensordot/concat/axis?
(model_5/second_dropout1/Tensordot/concatConcatV2/model_5/second_dropout1/Tensordot/free:output:0/model_5/second_dropout1/Tensordot/axes:output:06model_5/second_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(model_5/second_dropout1/Tensordot/concat?
'model_5/second_dropout1/Tensordot/stackPack/model_5/second_dropout1/Tensordot/Prod:output:01model_5/second_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'model_5/second_dropout1/Tensordot/stack?
+model_5/second_dropout1/Tensordot/transpose	Transpose.model_5/tf_op_layer_concat_5/concat_5:output:01model_5/second_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2-
+model_5/second_dropout1/Tensordot/transpose?
)model_5/second_dropout1/Tensordot/ReshapeReshape/model_5/second_dropout1/Tensordot/transpose:y:00model_5/second_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)model_5/second_dropout1/Tensordot/Reshape?
(model_5/second_dropout1/Tensordot/MatMulMatMul2model_5/second_dropout1/Tensordot/Reshape:output:08model_5/second_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(model_5/second_dropout1/Tensordot/MatMul?
)model_5/second_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2+
)model_5/second_dropout1/Tensordot/Const_2?
/model_5/second_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_5/second_dropout1/Tensordot/concat_1/axis?
*model_5/second_dropout1/Tensordot/concat_1ConcatV23model_5/second_dropout1/Tensordot/GatherV2:output:02model_5/second_dropout1/Tensordot/Const_2:output:08model_5/second_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*model_5/second_dropout1/Tensordot/concat_1?
!model_5/second_dropout1/TensordotReshape2model_5/second_dropout1/Tensordot/MatMul:product:03model_5/second_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2#
!model_5/second_dropout1/Tensordot?
.model_5/second_dropout1/BiasAdd/ReadVariableOpReadVariableOp7model_5_second_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.model_5/second_dropout1/BiasAdd/ReadVariableOp?
model_5/second_dropout1/BiasAddBiasAdd*model_5/second_dropout1/Tensordot:output:06model_5/second_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2!
model_5/second_dropout1/BiasAdd?
model_5/second_dropout1/SigmoidSigmoid(model_5/second_dropout1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2!
model_5/second_dropout1/Sigmoid?
-model_5/one_dropout1/Tensordot/ReadVariableOpReadVariableOp6model_5_one_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02/
-model_5/one_dropout1/Tensordot/ReadVariableOp?
#model_5/one_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#model_5/one_dropout1/Tensordot/axes?
#model_5/one_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_5/one_dropout1/Tensordot/free?
$model_5/one_dropout1/Tensordot/ShapeShape%model_5/interoutput/Identity:output:0*
T0*
_output_shapes
:2&
$model_5/one_dropout1/Tensordot/Shape?
,model_5/one_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_5/one_dropout1/Tensordot/GatherV2/axis?
'model_5/one_dropout1/Tensordot/GatherV2GatherV2-model_5/one_dropout1/Tensordot/Shape:output:0,model_5/one_dropout1/Tensordot/free:output:05model_5/one_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'model_5/one_dropout1/Tensordot/GatherV2?
.model_5/one_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_5/one_dropout1/Tensordot/GatherV2_1/axis?
)model_5/one_dropout1/Tensordot/GatherV2_1GatherV2-model_5/one_dropout1/Tensordot/Shape:output:0,model_5/one_dropout1/Tensordot/axes:output:07model_5/one_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)model_5/one_dropout1/Tensordot/GatherV2_1?
$model_5/one_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_5/one_dropout1/Tensordot/Const?
#model_5/one_dropout1/Tensordot/ProdProd0model_5/one_dropout1/Tensordot/GatherV2:output:0-model_5/one_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#model_5/one_dropout1/Tensordot/Prod?
&model_5/one_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/one_dropout1/Tensordot/Const_1?
%model_5/one_dropout1/Tensordot/Prod_1Prod2model_5/one_dropout1/Tensordot/GatherV2_1:output:0/model_5/one_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%model_5/one_dropout1/Tensordot/Prod_1?
*model_5/one_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_5/one_dropout1/Tensordot/concat/axis?
%model_5/one_dropout1/Tensordot/concatConcatV2,model_5/one_dropout1/Tensordot/free:output:0,model_5/one_dropout1/Tensordot/axes:output:03model_5/one_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%model_5/one_dropout1/Tensordot/concat?
$model_5/one_dropout1/Tensordot/stackPack,model_5/one_dropout1/Tensordot/Prod:output:0.model_5/one_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$model_5/one_dropout1/Tensordot/stack?
(model_5/one_dropout1/Tensordot/transpose	Transpose%model_5/interoutput/Identity:output:0.model_5/one_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2*
(model_5/one_dropout1/Tensordot/transpose?
&model_5/one_dropout1/Tensordot/ReshapeReshape,model_5/one_dropout1/Tensordot/transpose:y:0-model_5/one_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&model_5/one_dropout1/Tensordot/Reshape?
%model_5/one_dropout1/Tensordot/MatMulMatMul/model_5/one_dropout1/Tensordot/Reshape:output:05model_5/one_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%model_5/one_dropout1/Tensordot/MatMul?
&model_5/one_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&model_5/one_dropout1/Tensordot/Const_2?
,model_5/one_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_5/one_dropout1/Tensordot/concat_1/axis?
'model_5/one_dropout1/Tensordot/concat_1ConcatV20model_5/one_dropout1/Tensordot/GatherV2:output:0/model_5/one_dropout1/Tensordot/Const_2:output:05model_5/one_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'model_5/one_dropout1/Tensordot/concat_1?
model_5/one_dropout1/TensordotReshape/model_5/one_dropout1/Tensordot/MatMul:product:00model_5/one_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2 
model_5/one_dropout1/Tensordot?
+model_5/one_dropout1/BiasAdd/ReadVariableOpReadVariableOp4model_5_one_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model_5/one_dropout1/BiasAdd/ReadVariableOp?
model_5/one_dropout1/BiasAddBiasAdd'model_5/one_dropout1/Tensordot:output:03model_5/one_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
model_5/one_dropout1/BiasAdd?
*model_5/one_dropout1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_5/one_dropout1/Max/reduction_indices?
model_5/one_dropout1/MaxMax%model_5/one_dropout1/BiasAdd:output:03model_5/one_dropout1/Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
model_5/one_dropout1/Max?
model_5/one_dropout1/subSub%model_5/one_dropout1/BiasAdd:output:0!model_5/one_dropout1/Max:output:0*
T0*,
_output_shapes
:??????????@2
model_5/one_dropout1/sub?
model_5/one_dropout1/ExpExpmodel_5/one_dropout1/sub:z:0*
T0*,
_output_shapes
:??????????@2
model_5/one_dropout1/Exp?
*model_5/one_dropout1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_5/one_dropout1/Sum/reduction_indices?
model_5/one_dropout1/SumSummodel_5/one_dropout1/Exp:y:03model_5/one_dropout1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
model_5/one_dropout1/Sum?
model_5/one_dropout1/truedivRealDivmodel_5/one_dropout1/Exp:y:0!model_5/one_dropout1/Sum:output:0*
T0*,
_output_shapes
:??????????@2
model_5/one_dropout1/truediv?
model_5/reshape_17/ShapeShape#model_5/second_dropout1/Sigmoid:y:0*
T0*
_output_shapes
:2
model_5/reshape_17/Shape?
&model_5/reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/reshape_17/strided_slice/stack?
(model_5/reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_17/strided_slice/stack_1?
(model_5/reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_17/strided_slice/stack_2?
 model_5/reshape_17/strided_sliceStridedSlice!model_5/reshape_17/Shape:output:0/model_5/reshape_17/strided_slice/stack:output:01model_5/reshape_17/strided_slice/stack_1:output:01model_5/reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_5/reshape_17/strided_slice?
"model_5/reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_5/reshape_17/Reshape/shape/1?
"model_5/reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2$
"model_5/reshape_17/Reshape/shape/2?
 model_5/reshape_17/Reshape/shapePack)model_5/reshape_17/strided_slice:output:0+model_5/reshape_17/Reshape/shape/1:output:0+model_5/reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_5/reshape_17/Reshape/shape?
model_5/reshape_17/ReshapeReshape#model_5/second_dropout1/Sigmoid:y:0)model_5/reshape_17/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
model_5/reshape_17/Reshape?
model_5/reshape_15/ShapeShape model_5/one_dropout1/truediv:z:0*
T0*
_output_shapes
:2
model_5/reshape_15/Shape?
&model_5/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/reshape_15/strided_slice/stack?
(model_5/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_15/strided_slice/stack_1?
(model_5/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/reshape_15/strided_slice/stack_2?
 model_5/reshape_15/strided_sliceStridedSlice!model_5/reshape_15/Shape:output:0/model_5/reshape_15/strided_slice/stack:output:01model_5/reshape_15/strided_slice/stack_1:output:01model_5/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_5/reshape_15/strided_slice?
"model_5/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_5/reshape_15/Reshape/shape/1?
"model_5/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2$
"model_5/reshape_15/Reshape/shape/2?
 model_5/reshape_15/Reshape/shapePack)model_5/reshape_15/strided_slice:output:0+model_5/reshape_15/Reshape/shape/1:output:0+model_5/reshape_15/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 model_5/reshape_15/Reshape/shape?
model_5/reshape_15/ReshapeReshape model_5/one_dropout1/truediv:z:0)model_5/reshape_15/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
model_5/reshape_15/Reshape?
0model_5/second_dropout2/Tensordot/ReadVariableOpReadVariableOp9model_5_second_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype022
0model_5/second_dropout2/Tensordot/ReadVariableOp?
&model_5/second_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&model_5/second_dropout2/Tensordot/axes?
&model_5/second_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_5/second_dropout2/Tensordot/free?
'model_5/second_dropout2/Tensordot/ShapeShape#model_5/reshape_17/Reshape:output:0*
T0*
_output_shapes
:2)
'model_5/second_dropout2/Tensordot/Shape?
/model_5/second_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_5/second_dropout2/Tensordot/GatherV2/axis?
*model_5/second_dropout2/Tensordot/GatherV2GatherV20model_5/second_dropout2/Tensordot/Shape:output:0/model_5/second_dropout2/Tensordot/free:output:08model_5/second_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*model_5/second_dropout2/Tensordot/GatherV2?
1model_5/second_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1model_5/second_dropout2/Tensordot/GatherV2_1/axis?
,model_5/second_dropout2/Tensordot/GatherV2_1GatherV20model_5/second_dropout2/Tensordot/Shape:output:0/model_5/second_dropout2/Tensordot/axes:output:0:model_5/second_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,model_5/second_dropout2/Tensordot/GatherV2_1?
'model_5/second_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_5/second_dropout2/Tensordot/Const?
&model_5/second_dropout2/Tensordot/ProdProd3model_5/second_dropout2/Tensordot/GatherV2:output:00model_5/second_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&model_5/second_dropout2/Tensordot/Prod?
)model_5/second_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)model_5/second_dropout2/Tensordot/Const_1?
(model_5/second_dropout2/Tensordot/Prod_1Prod5model_5/second_dropout2/Tensordot/GatherV2_1:output:02model_5/second_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(model_5/second_dropout2/Tensordot/Prod_1?
-model_5/second_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_5/second_dropout2/Tensordot/concat/axis?
(model_5/second_dropout2/Tensordot/concatConcatV2/model_5/second_dropout2/Tensordot/free:output:0/model_5/second_dropout2/Tensordot/axes:output:06model_5/second_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(model_5/second_dropout2/Tensordot/concat?
'model_5/second_dropout2/Tensordot/stackPack/model_5/second_dropout2/Tensordot/Prod:output:01model_5/second_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'model_5/second_dropout2/Tensordot/stack?
+model_5/second_dropout2/Tensordot/transpose	Transpose#model_5/reshape_17/Reshape:output:01model_5/second_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2-
+model_5/second_dropout2/Tensordot/transpose?
)model_5/second_dropout2/Tensordot/ReshapeReshape/model_5/second_dropout2/Tensordot/transpose:y:00model_5/second_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)model_5/second_dropout2/Tensordot/Reshape?
(model_5/second_dropout2/Tensordot/MatMulMatMul2model_5/second_dropout2/Tensordot/Reshape:output:08model_5/second_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(model_5/second_dropout2/Tensordot/MatMul?
)model_5/second_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_5/second_dropout2/Tensordot/Const_2?
/model_5/second_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_5/second_dropout2/Tensordot/concat_1/axis?
*model_5/second_dropout2/Tensordot/concat_1ConcatV23model_5/second_dropout2/Tensordot/GatherV2:output:02model_5/second_dropout2/Tensordot/Const_2:output:08model_5/second_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*model_5/second_dropout2/Tensordot/concat_1?
!model_5/second_dropout2/TensordotReshape2model_5/second_dropout2/Tensordot/MatMul:product:03model_5/second_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2#
!model_5/second_dropout2/Tensordot?
.model_5/second_dropout2/BiasAdd/ReadVariableOpReadVariableOp7model_5_second_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model_5/second_dropout2/BiasAdd/ReadVariableOp?
model_5/second_dropout2/BiasAddBiasAdd*model_5/second_dropout2/Tensordot:output:06model_5/second_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2!
model_5/second_dropout2/BiasAdd?
model_5/second_dropout2/SigmoidSigmoid(model_5/second_dropout2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2!
model_5/second_dropout2/Sigmoid?
-model_5/one_dropout2/Tensordot/ReadVariableOpReadVariableOp6model_5_one_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02/
-model_5/one_dropout2/Tensordot/ReadVariableOp?
#model_5/one_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#model_5/one_dropout2/Tensordot/axes?
#model_5/one_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model_5/one_dropout2/Tensordot/free?
$model_5/one_dropout2/Tensordot/ShapeShape#model_5/reshape_15/Reshape:output:0*
T0*
_output_shapes
:2&
$model_5/one_dropout2/Tensordot/Shape?
,model_5/one_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_5/one_dropout2/Tensordot/GatherV2/axis?
'model_5/one_dropout2/Tensordot/GatherV2GatherV2-model_5/one_dropout2/Tensordot/Shape:output:0,model_5/one_dropout2/Tensordot/free:output:05model_5/one_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'model_5/one_dropout2/Tensordot/GatherV2?
.model_5/one_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_5/one_dropout2/Tensordot/GatherV2_1/axis?
)model_5/one_dropout2/Tensordot/GatherV2_1GatherV2-model_5/one_dropout2/Tensordot/Shape:output:0,model_5/one_dropout2/Tensordot/axes:output:07model_5/one_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)model_5/one_dropout2/Tensordot/GatherV2_1?
$model_5/one_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model_5/one_dropout2/Tensordot/Const?
#model_5/one_dropout2/Tensordot/ProdProd0model_5/one_dropout2/Tensordot/GatherV2:output:0-model_5/one_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#model_5/one_dropout2/Tensordot/Prod?
&model_5/one_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/one_dropout2/Tensordot/Const_1?
%model_5/one_dropout2/Tensordot/Prod_1Prod2model_5/one_dropout2/Tensordot/GatherV2_1:output:0/model_5/one_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%model_5/one_dropout2/Tensordot/Prod_1?
*model_5/one_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_5/one_dropout2/Tensordot/concat/axis?
%model_5/one_dropout2/Tensordot/concatConcatV2,model_5/one_dropout2/Tensordot/free:output:0,model_5/one_dropout2/Tensordot/axes:output:03model_5/one_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%model_5/one_dropout2/Tensordot/concat?
$model_5/one_dropout2/Tensordot/stackPack,model_5/one_dropout2/Tensordot/Prod:output:0.model_5/one_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$model_5/one_dropout2/Tensordot/stack?
(model_5/one_dropout2/Tensordot/transpose	Transpose#model_5/reshape_15/Reshape:output:0.model_5/one_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2*
(model_5/one_dropout2/Tensordot/transpose?
&model_5/one_dropout2/Tensordot/ReshapeReshape,model_5/one_dropout2/Tensordot/transpose:y:0-model_5/one_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&model_5/one_dropout2/Tensordot/Reshape?
%model_5/one_dropout2/Tensordot/MatMulMatMul/model_5/one_dropout2/Tensordot/Reshape:output:05model_5/one_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%model_5/one_dropout2/Tensordot/MatMul?
&model_5/one_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model_5/one_dropout2/Tensordot/Const_2?
,model_5/one_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_5/one_dropout2/Tensordot/concat_1/axis?
'model_5/one_dropout2/Tensordot/concat_1ConcatV20model_5/one_dropout2/Tensordot/GatherV2:output:0/model_5/one_dropout2/Tensordot/Const_2:output:05model_5/one_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'model_5/one_dropout2/Tensordot/concat_1?
model_5/one_dropout2/TensordotReshape/model_5/one_dropout2/Tensordot/MatMul:product:00model_5/one_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2 
model_5/one_dropout2/Tensordot?
+model_5/one_dropout2/BiasAdd/ReadVariableOpReadVariableOp4model_5_one_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_5/one_dropout2/BiasAdd/ReadVariableOp?
model_5/one_dropout2/BiasAddBiasAdd'model_5/one_dropout2/Tensordot:output:03model_5/one_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
model_5/one_dropout2/BiasAdd?
*model_5/one_dropout2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_5/one_dropout2/Max/reduction_indices?
model_5/one_dropout2/MaxMax%model_5/one_dropout2/BiasAdd:output:03model_5/one_dropout2/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_5/one_dropout2/Max?
model_5/one_dropout2/subSub%model_5/one_dropout2/BiasAdd:output:0!model_5/one_dropout2/Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_5/one_dropout2/sub?
model_5/one_dropout2/ExpExpmodel_5/one_dropout2/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
model_5/one_dropout2/Exp?
*model_5/one_dropout2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_5/one_dropout2/Sum/reduction_indices?
model_5/one_dropout2/SumSummodel_5/one_dropout2/Exp:y:03model_5/one_dropout2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_5/one_dropout2/Sum?
model_5/one_dropout2/truedivRealDivmodel_5/one_dropout2/Exp:y:0!model_5/one_dropout2/Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_5/one_dropout2/truediv?
model_5/second_output/ShapeShape#model_5/second_dropout2/Sigmoid:y:0*
T0*
_output_shapes
:2
model_5/second_output/Shape?
)model_5/second_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)model_5/second_output/strided_slice/stack?
+model_5/second_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_5/second_output/strided_slice/stack_1?
+model_5/second_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_5/second_output/strided_slice/stack_2?
#model_5/second_output/strided_sliceStridedSlice$model_5/second_output/Shape:output:02model_5/second_output/strided_slice/stack:output:04model_5/second_output/strided_slice/stack_1:output:04model_5/second_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#model_5/second_output/strided_slice?
%model_5/second_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model_5/second_output/Reshape/shape/1?
#model_5/second_output/Reshape/shapePack,model_5/second_output/strided_slice:output:0.model_5/second_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#model_5/second_output/Reshape/shape?
model_5/second_output/ReshapeReshape#model_5/second_dropout2/Sigmoid:y:0,model_5/second_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
model_5/second_output/Reshape?
model_5/one_output/ShapeShape model_5/one_dropout2/truediv:z:0*
T0*
_output_shapes
:2
model_5/one_output/Shape?
&model_5/one_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_5/one_output/strided_slice/stack?
(model_5/one_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/one_output/strided_slice/stack_1?
(model_5/one_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_5/one_output/strided_slice/stack_2?
 model_5/one_output/strided_sliceStridedSlice!model_5/one_output/Shape:output:0/model_5/one_output/strided_slice/stack:output:01model_5/one_output/strided_slice/stack_1:output:01model_5/one_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_5/one_output/strided_slice?
"model_5/one_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model_5/one_output/Reshape/shape/1?
 model_5/one_output/Reshape/shapePack)model_5/one_output/strided_slice:output:0+model_5/one_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 model_5/one_output/Reshape/shape?
model_5/one_output/ReshapeReshape model_5/one_dropout2/truediv:z:0)model_5/one_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
model_5/one_output/Reshape?
IdentityIdentity#model_5/one_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity?

Identity_1Identity&model_5/second_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
?
?
9__inference_batch_normalization_20_layer_call_fn_49604342

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
*__inference_model_5_layer_call_fn_49603811
input_69
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource@
<batch_normalization_20_batchnorm_mul_readvariableop_resource>
:batch_normalization_20_batchnorm_readvariableop_1_resource>
:batch_normalization_20_batchnorm_readvariableop_2_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource@
<batch_normalization_21_batchnorm_mul_readvariableop_resource>
:batch_normalization_21_batchnorm_readvariableop_1_resource>
:batch_normalization_21_batchnorm_readvariableop_2_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource<
8batch_normalization_22_batchnorm_readvariableop_resource@
<batch_normalization_22_batchnorm_mul_readvariableop_resource>
:batch_normalization_22_batchnorm_readvariableop_1_resource>
:batch_normalization_22_batchnorm_readvariableop_2_resource:
6conv1d_40_required_space_to_batch_paddings_block_shape9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource@
<batch_normalization_23_batchnorm_mul_readvariableop_resource>
:batch_normalization_23_batchnorm_readvariableop_1_resource>
:batch_normalization_23_batchnorm_readvariableop_2_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource5
1second_dropout1_tensordot_readvariableop_resource3
/second_dropout1_biasadd_readvariableop_resource2
.one_dropout1_tensordot_readvariableop_resource0
,one_dropout1_biasadd_readvariableop_resource5
1second_dropout2_tensordot_readvariableop_resource3
/second_dropout2_biasadd_readvariableop_resource2
.one_dropout2_tensordot_readvariableop_resource0
,one_dropout2_biasadd_readvariableop_resource
identity

identity_1??
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsinput_6(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1dConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/BiasAdd{
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/Relu?
max_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_15/ExpandDims/dim?
max_pooling1d_15/ExpandDims
ExpandDimsconv1d_35/Relu:activations:0(max_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_15/ExpandDims?
max_pooling1d_15/MaxPoolMaxPool$max_pooling1d_15/ExpandDims:output:0*0
_output_shapes
:??????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_15/MaxPool?
max_pooling1d_15/SqueezeSqueeze!max_pooling1d_15/MaxPool:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
max_pooling1d_15/Squeeze?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDims!max_pooling1d_15/Squeeze:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_36/BiasAdd|
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_36/Relu?
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim?
max_pooling1d_16/ExpandDims
ExpandDimsconv1d_36/Relu:activations:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_16/ExpandDims?
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool?
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_16/Squeeze?
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp?
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_20/batchnorm/add/y?
$batch_normalization_20/batchnorm/addAddV27batch_normalization_20/batchnorm/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/add?
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/Rsqrt?
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOp?
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/mul?
&batch_normalization_20/batchnorm/mul_1Mul!max_pooling1d_16/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/mul_1?
1batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_1?
&batch_normalization_20/batchnorm/mul_2Mul9batch_normalization_20/batchnorm/ReadVariableOp_1:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/mul_2?
1batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_2?
$batch_normalization_20/batchnorm/subSub9batch_normalization_20/batchnorm/ReadVariableOp_2:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/sub?
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/add_1?
activation_20/ReluRelu*batch_normalization_20/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_20/Relu?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDims activation_20/Relu:activations:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_37/BiasAdd?
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_37/conv1d_37/kernel/Regularizer/SquareSquareDconv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_37/conv1d_37/kernel/Regularizer/Square?
,conv1d_37/conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_37/conv1d_37/kernel/Regularizer/Const?
*conv1d_37/conv1d_37/kernel/Regularizer/SumSum1conv1d_37/conv1d_37/kernel/Regularizer/Square:y:05conv1d_37/conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/Sum?
,conv1d_37/conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_37/conv1d_37/kernel/Regularizer/mul/x?
*conv1d_37/conv1d_37/kernel/Regularizer/mulMul5conv1d_37/conv1d_37/kernel/Regularizer/mul/x:output:03conv1d_37/conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/mul?
,conv1d_37/conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_37/conv1d_37/kernel/Regularizer/add/x?
*conv1d_37/conv1d_37/kernel/Regularizer/addAddV25conv1d_37/conv1d_37/kernel/Regularizer/add/x:output:0.conv1d_37/conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/add?
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOp?
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_21/batchnorm/add/y?
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/add?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/Rsqrt?
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOp?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/mul?
&batch_normalization_21/batchnorm/mul_1Mulconv1d_37/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/mul_1?
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_1?
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/mul_2?
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_2?
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/sub?
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/add_1?
activation_21/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_21/Relu?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDims activation_21/Relu:activations:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_38/BiasAdd?
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_38/conv1d_38/kernel/Regularizer/SquareSquareDconv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_38/conv1d_38/kernel/Regularizer/Square?
,conv1d_38/conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_38/conv1d_38/kernel/Regularizer/Const?
*conv1d_38/conv1d_38/kernel/Regularizer/SumSum1conv1d_38/conv1d_38/kernel/Regularizer/Square:y:05conv1d_38/conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/Sum?
,conv1d_38/conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_38/conv1d_38/kernel/Regularizer/mul/x?
*conv1d_38/conv1d_38/kernel/Regularizer/mulMul5conv1d_38/conv1d_38/kernel/Regularizer/mul/x:output:03conv1d_38/conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/mul?
,conv1d_38/conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_38/conv1d_38/kernel/Regularizer/add/x?
*conv1d_38/conv1d_38/kernel/Regularizer/addAddV25conv1d_38/conv1d_38/kernel/Regularizer/add/x:output:0.conv1d_38/conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/add?

add_10/addAddV2conv1d_38/BiasAdd:output:0!max_pooling1d_16/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_10/add?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsadd_10/add:z:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_39/BiasAdd|
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_39/Relu?
max_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_17/ExpandDims/dim?
max_pooling1d_17/ExpandDims
ExpandDimsconv1d_39/Relu:activations:0(max_pooling1d_17/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_17/ExpandDims?
max_pooling1d_17/MaxPoolMaxPool$max_pooling1d_17/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_17/MaxPool?
max_pooling1d_17/SqueezeSqueeze!max_pooling1d_17/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_17/Squeeze?
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_22/batchnorm/ReadVariableOp?
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_22/batchnorm/add/y?
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/add?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/Rsqrt?
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_22/batchnorm/mul/ReadVariableOp?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/mul?
&batch_normalization_22/batchnorm/mul_1Mul!max_pooling1d_17/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/mul_1?
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_22/batchnorm/ReadVariableOp_1?
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/mul_2?
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_22/batchnorm/ReadVariableOp_2?
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/sub?
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/add_1?
activation_22/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_22/Relu?
6conv1d_40/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?28
6conv1d_40/required_space_to_batch_paddings/input_shape?
8conv1d_40/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8conv1d_40/required_space_to_batch_paddings/base_paddings?
3conv1d_40/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        25
3conv1d_40/required_space_to_batch_paddings/paddings?
0conv1d_40/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        22
0conv1d_40/required_space_to_batch_paddings/crops?
$conv1d_40/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/SpaceToBatchND/block_shape?
!conv1d_40/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2#
!conv1d_40/SpaceToBatchND/paddings?
conv1d_40/SpaceToBatchNDSpaceToBatchND activation_22/Relu:activations:0-conv1d_40/SpaceToBatchND/block_shape:output:0*conv1d_40/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:?????????A?2
conv1d_40/SpaceToBatchND?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDims!conv1d_40/SpaceToBatchND:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????A?2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1dConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????A?*
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*,
_output_shapes
:?????????A?*
squeeze_dims
2
conv1d_40/conv1d/Squeeze?
$conv1d_40/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/BatchToSpaceND/block_shape?
conv1d_40/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2 
conv1d_40/BatchToSpaceND/crops?
conv1d_40/BatchToSpaceNDBatchToSpaceND!conv1d_40/conv1d/Squeeze:output:0-conv1d_40/BatchToSpaceND/block_shape:output:0'conv1d_40/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BatchToSpaceND?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/BatchToSpaceND:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BiasAdd?
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_40/conv1d_40/kernel/Regularizer/SquareSquareDconv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_40/conv1d_40/kernel/Regularizer/Square?
,conv1d_40/conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_40/conv1d_40/kernel/Regularizer/Const?
*conv1d_40/conv1d_40/kernel/Regularizer/SumSum1conv1d_40/conv1d_40/kernel/Regularizer/Square:y:05conv1d_40/conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/Sum?
,conv1d_40/conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_40/conv1d_40/kernel/Regularizer/mul/x?
*conv1d_40/conv1d_40/kernel/Regularizer/mulMul5conv1d_40/conv1d_40/kernel/Regularizer/mul/x:output:03conv1d_40/conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/mul?
,conv1d_40/conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_40/conv1d_40/kernel/Regularizer/add/x?
*conv1d_40/conv1d_40/kernel/Regularizer/addAddV25conv1d_40/conv1d_40/kernel/Regularizer/add/x:output:0.conv1d_40/conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/add?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Mulconv1d_40/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/mul_1?
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_1?
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_2?
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/add_1?
activation_23/ReluRelu*batch_normalization_23/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_23/Relu?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDims activation_23/Relu:activations:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1dConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_41/BiasAdd?
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_41/conv1d_41/kernel/Regularizer/SquareSquareDconv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_41/conv1d_41/kernel/Regularizer/Square?
,conv1d_41/conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_41/conv1d_41/kernel/Regularizer/Const?
*conv1d_41/conv1d_41/kernel/Regularizer/SumSum1conv1d_41/conv1d_41/kernel/Regularizer/Square:y:05conv1d_41/conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/Sum?
,conv1d_41/conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_41/conv1d_41/kernel/Regularizer/mul/x?
*conv1d_41/conv1d_41/kernel/Regularizer/mulMul5conv1d_41/conv1d_41/kernel/Regularizer/mul/x:output:03conv1d_41/conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/mul?
,conv1d_41/conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_41/conv1d_41/kernel/Regularizer/add/x?
*conv1d_41/conv1d_41/kernel/Regularizer/addAddV25conv1d_41/conv1d_41/kernel/Regularizer/add/x:output:0.conv1d_41/conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/add?

add_11/addAddV2conv1d_41/BiasAdd:output:0!max_pooling1d_17/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_11/add?
interoutput/IdentityIdentityadd_11/add:z:0*
T0*-
_output_shapes
:???????????2
interoutput/Identity[
reshape_16/ShapeShapeinput_6*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice?
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinput_6!reshape_16/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
reshape_16/Reshape?
"tf_op_layer_concat_5/concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_5/concat_5/axis?
tf_op_layer_concat_5/concat_5ConcatV2interoutput/Identity:output:0reshape_16/Reshape:output:0+tf_op_layer_concat_5/concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2
tf_op_layer_concat_5/concat_5?
(second_dropout1/Tensordot/ReadVariableOpReadVariableOp1second_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(second_dropout1/Tensordot/ReadVariableOp?
second_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout1/Tensordot/axes?
second_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout1/Tensordot/free?
second_dropout1/Tensordot/ShapeShape&tf_op_layer_concat_5/concat_5:output:0*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/Shape?
'second_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/GatherV2/axis?
"second_dropout1/Tensordot/GatherV2GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/free:output:00second_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout1/Tensordot/GatherV2?
)second_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout1/Tensordot/GatherV2_1/axis?
$second_dropout1/Tensordot/GatherV2_1GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/axes:output:02second_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout1/Tensordot/GatherV2_1?
second_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout1/Tensordot/Const?
second_dropout1/Tensordot/ProdProd+second_dropout1/Tensordot/GatherV2:output:0(second_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout1/Tensordot/Prod?
!second_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout1/Tensordot/Const_1?
 second_dropout1/Tensordot/Prod_1Prod-second_dropout1/Tensordot/GatherV2_1:output:0*second_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout1/Tensordot/Prod_1?
%second_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout1/Tensordot/concat/axis?
 second_dropout1/Tensordot/concatConcatV2'second_dropout1/Tensordot/free:output:0'second_dropout1/Tensordot/axes:output:0.second_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout1/Tensordot/concat?
second_dropout1/Tensordot/stackPack'second_dropout1/Tensordot/Prod:output:0)second_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/stack?
#second_dropout1/Tensordot/transpose	Transpose&tf_op_layer_concat_5/concat_5:output:0)second_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2%
#second_dropout1/Tensordot/transpose?
!second_dropout1/Tensordot/ReshapeReshape'second_dropout1/Tensordot/transpose:y:0(second_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout1/Tensordot/Reshape?
 second_dropout1/Tensordot/MatMulMatMul*second_dropout1/Tensordot/Reshape:output:00second_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 second_dropout1/Tensordot/MatMul?
!second_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2#
!second_dropout1/Tensordot/Const_2?
'second_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/concat_1/axis?
"second_dropout1/Tensordot/concat_1ConcatV2+second_dropout1/Tensordot/GatherV2:output:0*second_dropout1/Tensordot/Const_2:output:00second_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout1/Tensordot/concat_1?
second_dropout1/TensordotReshape*second_dropout1/Tensordot/MatMul:product:0+second_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Tensordot?
&second_dropout1/BiasAdd/ReadVariableOpReadVariableOp/second_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&second_dropout1/BiasAdd/ReadVariableOp?
second_dropout1/BiasAddBiasAdd"second_dropout1/Tensordot:output:0.second_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/BiasAdd?
second_dropout1/SigmoidSigmoid second_dropout1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Sigmoid?
%one_dropout1/Tensordot/ReadVariableOpReadVariableOp.one_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%one_dropout1/Tensordot/ReadVariableOp?
one_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout1/Tensordot/axes?
one_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout1/Tensordot/free?
one_dropout1/Tensordot/ShapeShapeinteroutput/Identity:output:0*
T0*
_output_shapes
:2
one_dropout1/Tensordot/Shape?
$one_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/GatherV2/axis?
one_dropout1/Tensordot/GatherV2GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/free:output:0-one_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout1/Tensordot/GatherV2?
&one_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout1/Tensordot/GatherV2_1/axis?
!one_dropout1/Tensordot/GatherV2_1GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/axes:output:0/one_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout1/Tensordot/GatherV2_1?
one_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout1/Tensordot/Const?
one_dropout1/Tensordot/ProdProd(one_dropout1/Tensordot/GatherV2:output:0%one_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod?
one_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout1/Tensordot/Const_1?
one_dropout1/Tensordot/Prod_1Prod*one_dropout1/Tensordot/GatherV2_1:output:0'one_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod_1?
"one_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout1/Tensordot/concat/axis?
one_dropout1/Tensordot/concatConcatV2$one_dropout1/Tensordot/free:output:0$one_dropout1/Tensordot/axes:output:0+one_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/concat?
one_dropout1/Tensordot/stackPack$one_dropout1/Tensordot/Prod:output:0&one_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/stack?
 one_dropout1/Tensordot/transpose	Transposeinteroutput/Identity:output:0&one_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2"
 one_dropout1/Tensordot/transpose?
one_dropout1/Tensordot/ReshapeReshape$one_dropout1/Tensordot/transpose:y:0%one_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout1/Tensordot/Reshape?
one_dropout1/Tensordot/MatMulMatMul'one_dropout1/Tensordot/Reshape:output:0-one_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
one_dropout1/Tensordot/MatMul?
one_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2 
one_dropout1/Tensordot/Const_2?
$one_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/concat_1/axis?
one_dropout1/Tensordot/concat_1ConcatV2(one_dropout1/Tensordot/GatherV2:output:0'one_dropout1/Tensordot/Const_2:output:0-one_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout1/Tensordot/concat_1?
one_dropout1/TensordotReshape'one_dropout1/Tensordot/MatMul:product:0(one_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Tensordot?
#one_dropout1/BiasAdd/ReadVariableOpReadVariableOp,one_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#one_dropout1/BiasAdd/ReadVariableOp?
one_dropout1/BiasAddBiasAddone_dropout1/Tensordot:output:0+one_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/BiasAdd?
"one_dropout1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Max/reduction_indices?
one_dropout1/MaxMaxone_dropout1/BiasAdd:output:0+one_dropout1/Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Max?
one_dropout1/subSubone_dropout1/BiasAdd:output:0one_dropout1/Max:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/subx
one_dropout1/ExpExpone_dropout1/sub:z:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Exp?
"one_dropout1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Sum/reduction_indices?
one_dropout1/SumSumone_dropout1/Exp:y:0+one_dropout1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Sum?
one_dropout1/truedivRealDivone_dropout1/Exp:y:0one_dropout1/Sum:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/truedivo
reshape_17/ShapeShapesecond_dropout1/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slice?
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_17/Reshape/shape/1{
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapesecond_dropout1/Sigmoid:y:0!reshape_17/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_17/Reshapel
reshape_15/ShapeShapeone_dropout1/truediv:z:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slice?
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_15/Reshape/shape/1{
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_15/Reshape/shape/2?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeone_dropout1/truediv:z:0!reshape_15/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_15/Reshape?
(second_dropout2/Tensordot/ReadVariableOpReadVariableOp1second_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02*
(second_dropout2/Tensordot/ReadVariableOp?
second_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout2/Tensordot/axes?
second_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout2/Tensordot/free?
second_dropout2/Tensordot/ShapeShapereshape_17/Reshape:output:0*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/Shape?
'second_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/GatherV2/axis?
"second_dropout2/Tensordot/GatherV2GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/free:output:00second_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout2/Tensordot/GatherV2?
)second_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout2/Tensordot/GatherV2_1/axis?
$second_dropout2/Tensordot/GatherV2_1GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/axes:output:02second_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout2/Tensordot/GatherV2_1?
second_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout2/Tensordot/Const?
second_dropout2/Tensordot/ProdProd+second_dropout2/Tensordot/GatherV2:output:0(second_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout2/Tensordot/Prod?
!second_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout2/Tensordot/Const_1?
 second_dropout2/Tensordot/Prod_1Prod-second_dropout2/Tensordot/GatherV2_1:output:0*second_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout2/Tensordot/Prod_1?
%second_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout2/Tensordot/concat/axis?
 second_dropout2/Tensordot/concatConcatV2'second_dropout2/Tensordot/free:output:0'second_dropout2/Tensordot/axes:output:0.second_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout2/Tensordot/concat?
second_dropout2/Tensordot/stackPack'second_dropout2/Tensordot/Prod:output:0)second_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/stack?
#second_dropout2/Tensordot/transpose	Transposereshape_17/Reshape:output:0)second_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2%
#second_dropout2/Tensordot/transpose?
!second_dropout2/Tensordot/ReshapeReshape'second_dropout2/Tensordot/transpose:y:0(second_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout2/Tensordot/Reshape?
 second_dropout2/Tensordot/MatMulMatMul*second_dropout2/Tensordot/Reshape:output:00second_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 second_dropout2/Tensordot/MatMul?
!second_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!second_dropout2/Tensordot/Const_2?
'second_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/concat_1/axis?
"second_dropout2/Tensordot/concat_1ConcatV2+second_dropout2/Tensordot/GatherV2:output:0*second_dropout2/Tensordot/Const_2:output:00second_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout2/Tensordot/concat_1?
second_dropout2/TensordotReshape*second_dropout2/Tensordot/MatMul:product:0+second_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Tensordot?
&second_dropout2/BiasAdd/ReadVariableOpReadVariableOp/second_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&second_dropout2/BiasAdd/ReadVariableOp?
second_dropout2/BiasAddBiasAdd"second_dropout2/Tensordot:output:0.second_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/BiasAdd?
second_dropout2/SigmoidSigmoid second_dropout2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Sigmoid?
%one_dropout2/Tensordot/ReadVariableOpReadVariableOp.one_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02'
%one_dropout2/Tensordot/ReadVariableOp?
one_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout2/Tensordot/axes?
one_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout2/Tensordot/free?
one_dropout2/Tensordot/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
one_dropout2/Tensordot/Shape?
$one_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/GatherV2/axis?
one_dropout2/Tensordot/GatherV2GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/free:output:0-one_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout2/Tensordot/GatherV2?
&one_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout2/Tensordot/GatherV2_1/axis?
!one_dropout2/Tensordot/GatherV2_1GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/axes:output:0/one_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout2/Tensordot/GatherV2_1?
one_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout2/Tensordot/Const?
one_dropout2/Tensordot/ProdProd(one_dropout2/Tensordot/GatherV2:output:0%one_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod?
one_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout2/Tensordot/Const_1?
one_dropout2/Tensordot/Prod_1Prod*one_dropout2/Tensordot/GatherV2_1:output:0'one_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod_1?
"one_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout2/Tensordot/concat/axis?
one_dropout2/Tensordot/concatConcatV2$one_dropout2/Tensordot/free:output:0$one_dropout2/Tensordot/axes:output:0+one_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/concat?
one_dropout2/Tensordot/stackPack$one_dropout2/Tensordot/Prod:output:0&one_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/stack?
 one_dropout2/Tensordot/transpose	Transposereshape_15/Reshape:output:0&one_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2"
 one_dropout2/Tensordot/transpose?
one_dropout2/Tensordot/ReshapeReshape$one_dropout2/Tensordot/transpose:y:0%one_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout2/Tensordot/Reshape?
one_dropout2/Tensordot/MatMulMatMul'one_dropout2/Tensordot/Reshape:output:0-one_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
one_dropout2/Tensordot/MatMul?
one_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2 
one_dropout2/Tensordot/Const_2?
$one_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/concat_1/axis?
one_dropout2/Tensordot/concat_1ConcatV2(one_dropout2/Tensordot/GatherV2:output:0'one_dropout2/Tensordot/Const_2:output:0-one_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout2/Tensordot/concat_1?
one_dropout2/TensordotReshape'one_dropout2/Tensordot/MatMul:product:0(one_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Tensordot?
#one_dropout2/BiasAdd/ReadVariableOpReadVariableOp,one_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#one_dropout2/BiasAdd/ReadVariableOp?
one_dropout2/BiasAddBiasAddone_dropout2/Tensordot:output:0+one_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/BiasAdd?
"one_dropout2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Max/reduction_indices?
one_dropout2/MaxMaxone_dropout2/BiasAdd:output:0+one_dropout2/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Max?
one_dropout2/subSubone_dropout2/BiasAdd:output:0one_dropout2/Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/sub?
one_dropout2/ExpExpone_dropout2/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Exp?
"one_dropout2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Sum/reduction_indices?
one_dropout2/SumSumone_dropout2/Exp:y:0+one_dropout2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Sum?
one_dropout2/truedivRealDivone_dropout2/Exp:y:0one_dropout2/Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/truedivu
second_output/ShapeShapesecond_dropout2/Sigmoid:y:0*
T0*
_output_shapes
:2
second_output/Shape?
!second_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!second_output/strided_slice/stack?
#second_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_1?
#second_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_2?
second_output/strided_sliceStridedSlicesecond_output/Shape:output:0*second_output/strided_slice/stack:output:0,second_output/strided_slice/stack_1:output:0,second_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
second_output/strided_slice?
second_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
second_output/Reshape/shape/1?
second_output/Reshape/shapePack$second_output/strided_slice:output:0&second_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
second_output/Reshape/shape?
second_output/ReshapeReshapesecond_dropout2/Sigmoid:y:0$second_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
second_output/Reshapel
one_output/ShapeShapeone_dropout2/truediv:z:0*
T0*
_output_shapes
:2
one_output/Shape?
one_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
one_output/strided_slice/stack?
 one_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_1?
 one_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_2?
one_output/strided_sliceStridedSliceone_output/Shape:output:0'one_output/strided_slice/stack:output:0)one_output/strided_slice/stack_1:output:0)one_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
one_output/strided_slice?
one_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
one_output/Reshape/shape/1?
one_output/Reshape/shapePack!one_output/strided_slice:output:0#one_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
one_output/Reshape/shape?
one_output/ReshapeReshapeone_dropout2/truediv:z:0!one_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
one_output/Reshape?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/add?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/add?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/add?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/addx
IdentityIdentityone_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity

Identity_1Identitysecond_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
?
?
,__inference_conv1d_35_layer_call_fn_49600251

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_max_pooling1d_17_layer_call_fn_49600697

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_22_layer_call_fn_49604910

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_activation_23_layer_call_fn_49605198

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_activation_20_layer_call_and_return_conditional_losses_49604347

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
E__inference_model_5_layer_call_and_return_conditional_losses_49602515
input_69
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource3
/batch_normalization_20_assignmovingavg_496011995
1batch_normalization_20_assignmovingavg_1_49601205@
<batch_normalization_20_batchnorm_mul_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource3
/batch_normalization_21_assignmovingavg_496013755
1batch_normalization_21_assignmovingavg_1_49601381@
<batch_normalization_21_batchnorm_mul_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource3
/batch_normalization_22_assignmovingavg_496015825
1batch_normalization_22_assignmovingavg_1_49601588@
<batch_normalization_22_batchnorm_mul_readvariableop_resource<
8batch_normalization_22_batchnorm_readvariableop_resource:
6conv1d_40_required_space_to_batch_paddings_block_shape9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource3
/batch_normalization_23_assignmovingavg_496017695
1batch_normalization_23_assignmovingavg_1_49601775@
<batch_normalization_23_batchnorm_mul_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource5
1second_dropout1_tensordot_readvariableop_resource3
/second_dropout1_biasadd_readvariableop_resource2
.one_dropout1_tensordot_readvariableop_resource0
,one_dropout1_biasadd_readvariableop_resource5
1second_dropout2_tensordot_readvariableop_resource3
/second_dropout2_biasadd_readvariableop_resource2
.one_dropout2_tensordot_readvariableop_resource0
,one_dropout2_biasadd_readvariableop_resource
identity

identity_1??:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp?:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsinput_6(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1dConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/BiasAdd{
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/Relu?
max_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_15/ExpandDims/dim?
max_pooling1d_15/ExpandDims
ExpandDimsconv1d_35/Relu:activations:0(max_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_15/ExpandDims?
max_pooling1d_15/MaxPoolMaxPool$max_pooling1d_15/ExpandDims:output:0*0
_output_shapes
:??????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_15/MaxPool?
max_pooling1d_15/SqueezeSqueeze!max_pooling1d_15/MaxPool:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
max_pooling1d_15/Squeeze?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDims!max_pooling1d_15/Squeeze:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_36/BiasAdd|
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_36/Relu?
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim?
max_pooling1d_16/ExpandDims
ExpandDimsconv1d_36/Relu:activations:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_16/ExpandDims?
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool?
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_16/Squeeze?
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_20/moments/mean/reduction_indices?
#batch_normalization_20/moments/meanMean!max_pooling1d_16/Squeeze:output:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_20/moments/mean?
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_20/moments/StopGradient?
0batch_normalization_20/moments/SquaredDifferenceSquaredDifference!max_pooling1d_16/Squeeze:output:04batch_normalization_20/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_20/moments/SquaredDifference?
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_20/moments/variance/reduction_indices?
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_20/moments/variance?
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_20/moments/Squeeze?
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1?
,batch_normalization_20/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49601199*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_20/AssignMovingAvg/decay?
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_20_assignmovingavg_49601199*
_output_shapes	
:?*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOp?
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49601199*
_output_shapes	
:?2,
*batch_normalization_20/AssignMovingAvg/sub?
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:05batch_normalization_20/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49601199*
_output_shapes	
:?2,
*batch_normalization_20/AssignMovingAvg/mul?
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_20_assignmovingavg_49601199.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_20/AssignMovingAvg/49601199*
_output_shapes
 *
dtype02<
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_20/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49601205*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_20/AssignMovingAvg_1/decay?
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_20_assignmovingavg_1_49601205*
_output_shapes	
:?*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49601205*
_output_shapes	
:?2.
,batch_normalization_20/AssignMovingAvg_1/sub?
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:07batch_normalization_20/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49601205*
_output_shapes	
:?2.
,batch_normalization_20/AssignMovingAvg_1/mul?
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_20_assignmovingavg_1_496012050batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_20/AssignMovingAvg_1/49601205*
_output_shapes
 *
dtype02>
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_20/batchnorm/add/y?
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/add?
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/Rsqrt?
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOp?
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/mul?
&batch_normalization_20/batchnorm/mul_1Mul!max_pooling1d_16/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/mul_1?
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/mul_2?
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp?
$batch_normalization_20/batchnorm/subSub7batch_normalization_20/batchnorm/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/sub?
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/add_1?
activation_20/ReluRelu*batch_normalization_20/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_20/Relu?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDims activation_20/Relu:activations:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_37/BiasAdd?
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_37/conv1d_37/kernel/Regularizer/SquareSquareDconv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_37/conv1d_37/kernel/Regularizer/Square?
,conv1d_37/conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_37/conv1d_37/kernel/Regularizer/Const?
*conv1d_37/conv1d_37/kernel/Regularizer/SumSum1conv1d_37/conv1d_37/kernel/Regularizer/Square:y:05conv1d_37/conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/Sum?
,conv1d_37/conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_37/conv1d_37/kernel/Regularizer/mul/x?
*conv1d_37/conv1d_37/kernel/Regularizer/mulMul5conv1d_37/conv1d_37/kernel/Regularizer/mul/x:output:03conv1d_37/conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/mul?
,conv1d_37/conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_37/conv1d_37/kernel/Regularizer/add/x?
*conv1d_37/conv1d_37/kernel/Regularizer/addAddV25conv1d_37/conv1d_37/kernel/Regularizer/add/x:output:0.conv1d_37/conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/add?
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_21/moments/mean/reduction_indices?
#batch_normalization_21/moments/meanMeanconv1d_37/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_21/moments/mean?
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_21/moments/StopGradient?
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferenceconv1d_37/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_21/moments/SquaredDifference?
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_21/moments/variance/reduction_indices?
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_21/moments/variance?
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_21/moments/Squeeze?
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_21/moments/Squeeze_1?
,batch_normalization_21/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49601375*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_21/AssignMovingAvg/decay?
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_21_assignmovingavg_49601375*
_output_shapes	
:?*
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOp?
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49601375*
_output_shapes	
:?2,
*batch_normalization_21/AssignMovingAvg/sub?
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49601375*
_output_shapes	
:?2,
*batch_normalization_21/AssignMovingAvg/mul?
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_21_assignmovingavg_49601375.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_21/AssignMovingAvg/49601375*
_output_shapes
 *
dtype02<
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_21/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49601381*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_21/AssignMovingAvg_1/decay?
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_21_assignmovingavg_1_49601381*
_output_shapes	
:?*
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49601381*
_output_shapes	
:?2.
,batch_normalization_21/AssignMovingAvg_1/sub?
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49601381*
_output_shapes	
:?2.
,batch_normalization_21/AssignMovingAvg_1/mul?
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_21_assignmovingavg_1_496013810batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_21/AssignMovingAvg_1/49601381*
_output_shapes
 *
dtype02>
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_21/batchnorm/add/y?
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/add?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/Rsqrt?
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOp?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/mul?
&batch_normalization_21/batchnorm/mul_1Mulconv1d_37/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/mul_1?
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/mul_2?
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOp?
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/sub?
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/add_1?
activation_21/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_21/Relu?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDims activation_21/Relu:activations:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_38/BiasAdd?
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_38/conv1d_38/kernel/Regularizer/SquareSquareDconv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_38/conv1d_38/kernel/Regularizer/Square?
,conv1d_38/conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_38/conv1d_38/kernel/Regularizer/Const?
*conv1d_38/conv1d_38/kernel/Regularizer/SumSum1conv1d_38/conv1d_38/kernel/Regularizer/Square:y:05conv1d_38/conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/Sum?
,conv1d_38/conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_38/conv1d_38/kernel/Regularizer/mul/x?
*conv1d_38/conv1d_38/kernel/Regularizer/mulMul5conv1d_38/conv1d_38/kernel/Regularizer/mul/x:output:03conv1d_38/conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/mul?
,conv1d_38/conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_38/conv1d_38/kernel/Regularizer/add/x?
*conv1d_38/conv1d_38/kernel/Regularizer/addAddV25conv1d_38/conv1d_38/kernel/Regularizer/add/x:output:0.conv1d_38/conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/add?

add_10/addAddV2conv1d_38/BiasAdd:output:0!max_pooling1d_16/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_10/add?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsadd_10/add:z:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_39/BiasAdd|
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_39/Relu?
max_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_17/ExpandDims/dim?
max_pooling1d_17/ExpandDims
ExpandDimsconv1d_39/Relu:activations:0(max_pooling1d_17/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_17/ExpandDims?
max_pooling1d_17/MaxPoolMaxPool$max_pooling1d_17/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_17/MaxPool?
max_pooling1d_17/SqueezeSqueeze!max_pooling1d_17/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_17/Squeeze?
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_22/moments/mean/reduction_indices?
#batch_normalization_22/moments/meanMean!max_pooling1d_17/Squeeze:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_22/moments/mean?
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_22/moments/StopGradient?
0batch_normalization_22/moments/SquaredDifferenceSquaredDifference!max_pooling1d_17/Squeeze:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_22/moments/SquaredDifference?
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_22/moments/variance/reduction_indices?
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_22/moments/variance?
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_22/moments/Squeeze?
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_22/moments/Squeeze_1?
,batch_normalization_22/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49601582*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_22/AssignMovingAvg/decay?
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_22_assignmovingavg_49601582*
_output_shapes	
:?*
dtype027
5batch_normalization_22/AssignMovingAvg/ReadVariableOp?
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49601582*
_output_shapes	
:?2,
*batch_normalization_22/AssignMovingAvg/sub?
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49601582*
_output_shapes	
:?2,
*batch_normalization_22/AssignMovingAvg/mul?
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_22_assignmovingavg_49601582.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_22/AssignMovingAvg/49601582*
_output_shapes
 *
dtype02<
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_22/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49601588*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_22/AssignMovingAvg_1/decay?
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_22_assignmovingavg_1_49601588*
_output_shapes	
:?*
dtype029
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49601588*
_output_shapes	
:?2.
,batch_normalization_22/AssignMovingAvg_1/sub?
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49601588*
_output_shapes	
:?2.
,batch_normalization_22/AssignMovingAvg_1/mul?
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_22_assignmovingavg_1_496015880batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_22/AssignMovingAvg_1/49601588*
_output_shapes
 *
dtype02>
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_22/batchnorm/add/y?
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/add?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/Rsqrt?
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_22/batchnorm/mul/ReadVariableOp?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/mul?
&batch_normalization_22/batchnorm/mul_1Mul!max_pooling1d_17/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/mul_1?
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/mul_2?
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_22/batchnorm/ReadVariableOp?
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/sub?
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/add_1?
activation_22/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_22/Relu?
6conv1d_40/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?28
6conv1d_40/required_space_to_batch_paddings/input_shape?
8conv1d_40/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8conv1d_40/required_space_to_batch_paddings/base_paddings?
3conv1d_40/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        25
3conv1d_40/required_space_to_batch_paddings/paddings?
0conv1d_40/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        22
0conv1d_40/required_space_to_batch_paddings/crops?
$conv1d_40/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/SpaceToBatchND/block_shape?
!conv1d_40/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2#
!conv1d_40/SpaceToBatchND/paddings?
conv1d_40/SpaceToBatchNDSpaceToBatchND activation_22/Relu:activations:0-conv1d_40/SpaceToBatchND/block_shape:output:0*conv1d_40/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:?????????A?2
conv1d_40/SpaceToBatchND?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDims!conv1d_40/SpaceToBatchND:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????A?2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1dConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????A?*
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*,
_output_shapes
:?????????A?*
squeeze_dims
2
conv1d_40/conv1d/Squeeze?
$conv1d_40/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/BatchToSpaceND/block_shape?
conv1d_40/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2 
conv1d_40/BatchToSpaceND/crops?
conv1d_40/BatchToSpaceNDBatchToSpaceND!conv1d_40/conv1d/Squeeze:output:0-conv1d_40/BatchToSpaceND/block_shape:output:0'conv1d_40/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BatchToSpaceND?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/BatchToSpaceND:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BiasAdd?
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_40/conv1d_40/kernel/Regularizer/SquareSquareDconv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_40/conv1d_40/kernel/Regularizer/Square?
,conv1d_40/conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_40/conv1d_40/kernel/Regularizer/Const?
*conv1d_40/conv1d_40/kernel/Regularizer/SumSum1conv1d_40/conv1d_40/kernel/Regularizer/Square:y:05conv1d_40/conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/Sum?
,conv1d_40/conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_40/conv1d_40/kernel/Regularizer/mul/x?
*conv1d_40/conv1d_40/kernel/Regularizer/mulMul5conv1d_40/conv1d_40/kernel/Regularizer/mul/x:output:03conv1d_40/conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/mul?
,conv1d_40/conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_40/conv1d_40/kernel/Regularizer/add/x?
*conv1d_40/conv1d_40/kernel/Regularizer/addAddV25conv1d_40/conv1d_40/kernel/Regularizer/add/x:output:0.conv1d_40/conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/add?
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_23/moments/mean/reduction_indices?
#batch_normalization_23/moments/meanMeanconv1d_40/BiasAdd:output:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2%
#batch_normalization_23/moments/mean?
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*#
_output_shapes
:?2-
+batch_normalization_23/moments/StopGradient?
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferenceconv1d_40/BiasAdd:output:04batch_normalization_23/moments/StopGradient:output:0*
T0*-
_output_shapes
:???????????22
0batch_normalization_23/moments/SquaredDifference?
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_23/moments/variance/reduction_indices?
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2)
'batch_normalization_23/moments/variance?
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_23/moments/Squeeze?
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_23/moments/Squeeze_1?
,batch_normalization_23/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49601769*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_23/AssignMovingAvg/decay?
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_23_assignmovingavg_49601769*
_output_shapes	
:?*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOp?
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49601769*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/sub?
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49601769*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/mul?
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_23_assignmovingavg_49601769.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/49601769*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_23/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49601775*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_23/AssignMovingAvg_1/decay?
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_23_assignmovingavg_1_49601775*
_output_shapes	
:?*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49601775*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/sub?
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49601775*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/mul?
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_23_assignmovingavg_1_496017750batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/49601775*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Mulconv1d_40/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/mul_1?
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/add_1?
activation_23/ReluRelu*batch_normalization_23/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_23/Relu?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDims activation_23/Relu:activations:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1dConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_41/BiasAdd?
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_41/conv1d_41/kernel/Regularizer/SquareSquareDconv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_41/conv1d_41/kernel/Regularizer/Square?
,conv1d_41/conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_41/conv1d_41/kernel/Regularizer/Const?
*conv1d_41/conv1d_41/kernel/Regularizer/SumSum1conv1d_41/conv1d_41/kernel/Regularizer/Square:y:05conv1d_41/conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/Sum?
,conv1d_41/conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_41/conv1d_41/kernel/Regularizer/mul/x?
*conv1d_41/conv1d_41/kernel/Regularizer/mulMul5conv1d_41/conv1d_41/kernel/Regularizer/mul/x:output:03conv1d_41/conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/mul?
,conv1d_41/conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_41/conv1d_41/kernel/Regularizer/add/x?
*conv1d_41/conv1d_41/kernel/Regularizer/addAddV25conv1d_41/conv1d_41/kernel/Regularizer/add/x:output:0.conv1d_41/conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/add?

add_11/addAddV2conv1d_41/BiasAdd:output:0!max_pooling1d_17/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_11/add{
interoutput/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *O???2
interoutput/dropout/Const?
interoutput/dropout/MulMuladd_11/add:z:0"interoutput/dropout/Const:output:0*
T0*-
_output_shapes
:???????????2
interoutput/dropout/Mult
interoutput/dropout/ShapeShapeadd_11/add:z:0*
T0*
_output_shapes
:2
interoutput/dropout/Shape?
0interoutput/dropout/random_uniform/RandomUniformRandomUniform"interoutput/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype022
0interoutput/dropout/random_uniform/RandomUniform?
"interoutput/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *33?>2$
"interoutput/dropout/GreaterEqual/y?
 interoutput/dropout/GreaterEqualGreaterEqual9interoutput/dropout/random_uniform/RandomUniform:output:0+interoutput/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2"
 interoutput/dropout/GreaterEqual?
interoutput/dropout/CastCast$interoutput/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
interoutput/dropout/Cast?
interoutput/dropout/Mul_1Mulinteroutput/dropout/Mul:z:0interoutput/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
interoutput/dropout/Mul_1[
reshape_16/ShapeShapeinput_6*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice?
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinput_6!reshape_16/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
reshape_16/Reshape?
"tf_op_layer_concat_5/concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_5/concat_5/axis?
tf_op_layer_concat_5/concat_5ConcatV2interoutput/dropout/Mul_1:z:0reshape_16/Reshape:output:0+tf_op_layer_concat_5/concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2
tf_op_layer_concat_5/concat_5?
(second_dropout1/Tensordot/ReadVariableOpReadVariableOp1second_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(second_dropout1/Tensordot/ReadVariableOp?
second_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout1/Tensordot/axes?
second_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout1/Tensordot/free?
second_dropout1/Tensordot/ShapeShape&tf_op_layer_concat_5/concat_5:output:0*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/Shape?
'second_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/GatherV2/axis?
"second_dropout1/Tensordot/GatherV2GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/free:output:00second_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout1/Tensordot/GatherV2?
)second_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout1/Tensordot/GatherV2_1/axis?
$second_dropout1/Tensordot/GatherV2_1GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/axes:output:02second_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout1/Tensordot/GatherV2_1?
second_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout1/Tensordot/Const?
second_dropout1/Tensordot/ProdProd+second_dropout1/Tensordot/GatherV2:output:0(second_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout1/Tensordot/Prod?
!second_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout1/Tensordot/Const_1?
 second_dropout1/Tensordot/Prod_1Prod-second_dropout1/Tensordot/GatherV2_1:output:0*second_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout1/Tensordot/Prod_1?
%second_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout1/Tensordot/concat/axis?
 second_dropout1/Tensordot/concatConcatV2'second_dropout1/Tensordot/free:output:0'second_dropout1/Tensordot/axes:output:0.second_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout1/Tensordot/concat?
second_dropout1/Tensordot/stackPack'second_dropout1/Tensordot/Prod:output:0)second_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/stack?
#second_dropout1/Tensordot/transpose	Transpose&tf_op_layer_concat_5/concat_5:output:0)second_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2%
#second_dropout1/Tensordot/transpose?
!second_dropout1/Tensordot/ReshapeReshape'second_dropout1/Tensordot/transpose:y:0(second_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout1/Tensordot/Reshape?
 second_dropout1/Tensordot/MatMulMatMul*second_dropout1/Tensordot/Reshape:output:00second_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 second_dropout1/Tensordot/MatMul?
!second_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2#
!second_dropout1/Tensordot/Const_2?
'second_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/concat_1/axis?
"second_dropout1/Tensordot/concat_1ConcatV2+second_dropout1/Tensordot/GatherV2:output:0*second_dropout1/Tensordot/Const_2:output:00second_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout1/Tensordot/concat_1?
second_dropout1/TensordotReshape*second_dropout1/Tensordot/MatMul:product:0+second_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Tensordot?
&second_dropout1/BiasAdd/ReadVariableOpReadVariableOp/second_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&second_dropout1/BiasAdd/ReadVariableOp?
second_dropout1/BiasAddBiasAdd"second_dropout1/Tensordot:output:0.second_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/BiasAdd?
second_dropout1/SigmoidSigmoid second_dropout1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Sigmoid?
%one_dropout1/Tensordot/ReadVariableOpReadVariableOp.one_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%one_dropout1/Tensordot/ReadVariableOp?
one_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout1/Tensordot/axes?
one_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout1/Tensordot/free?
one_dropout1/Tensordot/ShapeShapeinteroutput/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
one_dropout1/Tensordot/Shape?
$one_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/GatherV2/axis?
one_dropout1/Tensordot/GatherV2GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/free:output:0-one_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout1/Tensordot/GatherV2?
&one_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout1/Tensordot/GatherV2_1/axis?
!one_dropout1/Tensordot/GatherV2_1GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/axes:output:0/one_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout1/Tensordot/GatherV2_1?
one_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout1/Tensordot/Const?
one_dropout1/Tensordot/ProdProd(one_dropout1/Tensordot/GatherV2:output:0%one_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod?
one_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout1/Tensordot/Const_1?
one_dropout1/Tensordot/Prod_1Prod*one_dropout1/Tensordot/GatherV2_1:output:0'one_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod_1?
"one_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout1/Tensordot/concat/axis?
one_dropout1/Tensordot/concatConcatV2$one_dropout1/Tensordot/free:output:0$one_dropout1/Tensordot/axes:output:0+one_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/concat?
one_dropout1/Tensordot/stackPack$one_dropout1/Tensordot/Prod:output:0&one_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/stack?
 one_dropout1/Tensordot/transpose	Transposeinteroutput/dropout/Mul_1:z:0&one_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2"
 one_dropout1/Tensordot/transpose?
one_dropout1/Tensordot/ReshapeReshape$one_dropout1/Tensordot/transpose:y:0%one_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout1/Tensordot/Reshape?
one_dropout1/Tensordot/MatMulMatMul'one_dropout1/Tensordot/Reshape:output:0-one_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
one_dropout1/Tensordot/MatMul?
one_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2 
one_dropout1/Tensordot/Const_2?
$one_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/concat_1/axis?
one_dropout1/Tensordot/concat_1ConcatV2(one_dropout1/Tensordot/GatherV2:output:0'one_dropout1/Tensordot/Const_2:output:0-one_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout1/Tensordot/concat_1?
one_dropout1/TensordotReshape'one_dropout1/Tensordot/MatMul:product:0(one_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Tensordot?
#one_dropout1/BiasAdd/ReadVariableOpReadVariableOp,one_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#one_dropout1/BiasAdd/ReadVariableOp?
one_dropout1/BiasAddBiasAddone_dropout1/Tensordot:output:0+one_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/BiasAdd?
"one_dropout1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Max/reduction_indices?
one_dropout1/MaxMaxone_dropout1/BiasAdd:output:0+one_dropout1/Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Max?
one_dropout1/subSubone_dropout1/BiasAdd:output:0one_dropout1/Max:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/subx
one_dropout1/ExpExpone_dropout1/sub:z:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Exp?
"one_dropout1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Sum/reduction_indices?
one_dropout1/SumSumone_dropout1/Exp:y:0+one_dropout1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Sum?
one_dropout1/truedivRealDivone_dropout1/Exp:y:0one_dropout1/Sum:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/truedivo
reshape_17/ShapeShapesecond_dropout1/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slice?
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_17/Reshape/shape/1{
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapesecond_dropout1/Sigmoid:y:0!reshape_17/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_17/Reshapel
reshape_15/ShapeShapeone_dropout1/truediv:z:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slice?
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_15/Reshape/shape/1{
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_15/Reshape/shape/2?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeone_dropout1/truediv:z:0!reshape_15/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_15/Reshape?
(second_dropout2/Tensordot/ReadVariableOpReadVariableOp1second_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02*
(second_dropout2/Tensordot/ReadVariableOp?
second_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout2/Tensordot/axes?
second_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout2/Tensordot/free?
second_dropout2/Tensordot/ShapeShapereshape_17/Reshape:output:0*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/Shape?
'second_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/GatherV2/axis?
"second_dropout2/Tensordot/GatherV2GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/free:output:00second_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout2/Tensordot/GatherV2?
)second_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout2/Tensordot/GatherV2_1/axis?
$second_dropout2/Tensordot/GatherV2_1GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/axes:output:02second_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout2/Tensordot/GatherV2_1?
second_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout2/Tensordot/Const?
second_dropout2/Tensordot/ProdProd+second_dropout2/Tensordot/GatherV2:output:0(second_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout2/Tensordot/Prod?
!second_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout2/Tensordot/Const_1?
 second_dropout2/Tensordot/Prod_1Prod-second_dropout2/Tensordot/GatherV2_1:output:0*second_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout2/Tensordot/Prod_1?
%second_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout2/Tensordot/concat/axis?
 second_dropout2/Tensordot/concatConcatV2'second_dropout2/Tensordot/free:output:0'second_dropout2/Tensordot/axes:output:0.second_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout2/Tensordot/concat?
second_dropout2/Tensordot/stackPack'second_dropout2/Tensordot/Prod:output:0)second_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/stack?
#second_dropout2/Tensordot/transpose	Transposereshape_17/Reshape:output:0)second_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2%
#second_dropout2/Tensordot/transpose?
!second_dropout2/Tensordot/ReshapeReshape'second_dropout2/Tensordot/transpose:y:0(second_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout2/Tensordot/Reshape?
 second_dropout2/Tensordot/MatMulMatMul*second_dropout2/Tensordot/Reshape:output:00second_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 second_dropout2/Tensordot/MatMul?
!second_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!second_dropout2/Tensordot/Const_2?
'second_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/concat_1/axis?
"second_dropout2/Tensordot/concat_1ConcatV2+second_dropout2/Tensordot/GatherV2:output:0*second_dropout2/Tensordot/Const_2:output:00second_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout2/Tensordot/concat_1?
second_dropout2/TensordotReshape*second_dropout2/Tensordot/MatMul:product:0+second_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Tensordot?
&second_dropout2/BiasAdd/ReadVariableOpReadVariableOp/second_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&second_dropout2/BiasAdd/ReadVariableOp?
second_dropout2/BiasAddBiasAdd"second_dropout2/Tensordot:output:0.second_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/BiasAdd?
second_dropout2/SigmoidSigmoid second_dropout2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Sigmoid?
%one_dropout2/Tensordot/ReadVariableOpReadVariableOp.one_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02'
%one_dropout2/Tensordot/ReadVariableOp?
one_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout2/Tensordot/axes?
one_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout2/Tensordot/free?
one_dropout2/Tensordot/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
one_dropout2/Tensordot/Shape?
$one_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/GatherV2/axis?
one_dropout2/Tensordot/GatherV2GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/free:output:0-one_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout2/Tensordot/GatherV2?
&one_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout2/Tensordot/GatherV2_1/axis?
!one_dropout2/Tensordot/GatherV2_1GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/axes:output:0/one_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout2/Tensordot/GatherV2_1?
one_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout2/Tensordot/Const?
one_dropout2/Tensordot/ProdProd(one_dropout2/Tensordot/GatherV2:output:0%one_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod?
one_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout2/Tensordot/Const_1?
one_dropout2/Tensordot/Prod_1Prod*one_dropout2/Tensordot/GatherV2_1:output:0'one_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod_1?
"one_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout2/Tensordot/concat/axis?
one_dropout2/Tensordot/concatConcatV2$one_dropout2/Tensordot/free:output:0$one_dropout2/Tensordot/axes:output:0+one_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/concat?
one_dropout2/Tensordot/stackPack$one_dropout2/Tensordot/Prod:output:0&one_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/stack?
 one_dropout2/Tensordot/transpose	Transposereshape_15/Reshape:output:0&one_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2"
 one_dropout2/Tensordot/transpose?
one_dropout2/Tensordot/ReshapeReshape$one_dropout2/Tensordot/transpose:y:0%one_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout2/Tensordot/Reshape?
one_dropout2/Tensordot/MatMulMatMul'one_dropout2/Tensordot/Reshape:output:0-one_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
one_dropout2/Tensordot/MatMul?
one_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2 
one_dropout2/Tensordot/Const_2?
$one_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/concat_1/axis?
one_dropout2/Tensordot/concat_1ConcatV2(one_dropout2/Tensordot/GatherV2:output:0'one_dropout2/Tensordot/Const_2:output:0-one_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout2/Tensordot/concat_1?
one_dropout2/TensordotReshape'one_dropout2/Tensordot/MatMul:product:0(one_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Tensordot?
#one_dropout2/BiasAdd/ReadVariableOpReadVariableOp,one_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#one_dropout2/BiasAdd/ReadVariableOp?
one_dropout2/BiasAddBiasAddone_dropout2/Tensordot:output:0+one_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/BiasAdd?
"one_dropout2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Max/reduction_indices?
one_dropout2/MaxMaxone_dropout2/BiasAdd:output:0+one_dropout2/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Max?
one_dropout2/subSubone_dropout2/BiasAdd:output:0one_dropout2/Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/sub?
one_dropout2/ExpExpone_dropout2/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Exp?
"one_dropout2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Sum/reduction_indices?
one_dropout2/SumSumone_dropout2/Exp:y:0+one_dropout2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Sum?
one_dropout2/truedivRealDivone_dropout2/Exp:y:0one_dropout2/Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/truedivu
second_output/ShapeShapesecond_dropout2/Sigmoid:y:0*
T0*
_output_shapes
:2
second_output/Shape?
!second_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!second_output/strided_slice/stack?
#second_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_1?
#second_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_2?
second_output/strided_sliceStridedSlicesecond_output/Shape:output:0*second_output/strided_slice/stack:output:0,second_output/strided_slice/stack_1:output:0,second_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
second_output/strided_slice?
second_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
second_output/Reshape/shape/1?
second_output/Reshape/shapePack$second_output/strided_slice:output:0&second_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
second_output/Reshape/shape?
second_output/ReshapeReshapesecond_dropout2/Sigmoid:y:0$second_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
second_output/Reshapel
one_output/ShapeShapeone_dropout2/truediv:z:0*
T0*
_output_shapes
:2
one_output/Shape?
one_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
one_output/strided_slice/stack?
 one_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_1?
 one_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_2?
one_output/strided_sliceStridedSliceone_output/Shape:output:0'one_output/strided_slice/stack:output:0)one_output/strided_slice/stack_1:output:0)one_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
one_output/strided_slice?
one_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
one_output/Reshape/shape/1?
one_output/Reshape/shapePack!one_output/strided_slice:output:0#one_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
one_output/Reshape/shape?
one_output/ReshapeReshapeone_dropout2/truediv:z:0!one_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
one_output/Reshape?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/add?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/add?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/add?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/add?
IdentityIdentityone_output/Reshape:output:0;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:??????????????????2

Identity?

Identity_1Identitysecond_output/Reshape:output:0;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????:::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
??
?
E__inference_model_5_layer_call_and_return_conditional_losses_49602923
input_69
5conv1d_35_conv1d_expanddims_1_readvariableop_resource-
)conv1d_35_biasadd_readvariableop_resource9
5conv1d_36_conv1d_expanddims_1_readvariableop_resource-
)conv1d_36_biasadd_readvariableop_resource<
8batch_normalization_20_batchnorm_readvariableop_resource@
<batch_normalization_20_batchnorm_mul_readvariableop_resource>
:batch_normalization_20_batchnorm_readvariableop_1_resource>
:batch_normalization_20_batchnorm_readvariableop_2_resource9
5conv1d_37_conv1d_expanddims_1_readvariableop_resource-
)conv1d_37_biasadd_readvariableop_resource<
8batch_normalization_21_batchnorm_readvariableop_resource@
<batch_normalization_21_batchnorm_mul_readvariableop_resource>
:batch_normalization_21_batchnorm_readvariableop_1_resource>
:batch_normalization_21_batchnorm_readvariableop_2_resource9
5conv1d_38_conv1d_expanddims_1_readvariableop_resource-
)conv1d_38_biasadd_readvariableop_resource9
5conv1d_39_conv1d_expanddims_1_readvariableop_resource-
)conv1d_39_biasadd_readvariableop_resource<
8batch_normalization_22_batchnorm_readvariableop_resource@
<batch_normalization_22_batchnorm_mul_readvariableop_resource>
:batch_normalization_22_batchnorm_readvariableop_1_resource>
:batch_normalization_22_batchnorm_readvariableop_2_resource:
6conv1d_40_required_space_to_batch_paddings_block_shape9
5conv1d_40_conv1d_expanddims_1_readvariableop_resource-
)conv1d_40_biasadd_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource@
<batch_normalization_23_batchnorm_mul_readvariableop_resource>
:batch_normalization_23_batchnorm_readvariableop_1_resource>
:batch_normalization_23_batchnorm_readvariableop_2_resource9
5conv1d_41_conv1d_expanddims_1_readvariableop_resource-
)conv1d_41_biasadd_readvariableop_resource5
1second_dropout1_tensordot_readvariableop_resource3
/second_dropout1_biasadd_readvariableop_resource2
.one_dropout1_tensordot_readvariableop_resource0
,one_dropout1_biasadd_readvariableop_resource5
1second_dropout2_tensordot_readvariableop_resource3
/second_dropout2_biasadd_readvariableop_resource2
.one_dropout2_tensordot_readvariableop_resource0
,one_dropout2_biasadd_readvariableop_resource
identity

identity_1??
conv1d_35/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_35/conv1d/ExpandDims/dim?
conv1d_35/conv1d/ExpandDims
ExpandDimsinput_6(conv1d_35/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_35/conv1d/ExpandDims?
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_35_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02.
,conv1d_35/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_35/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_35/conv1d/ExpandDims_1/dim?
conv1d_35/conv1d/ExpandDims_1
ExpandDims4conv1d_35/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_35/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_35/conv1d/ExpandDims_1?
conv1d_35/conv1dConv2D$conv1d_35/conv1d/ExpandDims:output:0&conv1d_35/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_35/conv1d?
conv1d_35/conv1d/SqueezeSqueezeconv1d_35/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
conv1d_35/conv1d/Squeeze?
 conv1d_35/BiasAdd/ReadVariableOpReadVariableOp)conv1d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv1d_35/BiasAdd/ReadVariableOp?
conv1d_35/BiasAddBiasAdd!conv1d_35/conv1d/Squeeze:output:0(conv1d_35/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/BiasAdd{
conv1d_35/ReluReluconv1d_35/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_35/Relu?
max_pooling1d_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_15/ExpandDims/dim?
max_pooling1d_15/ExpandDims
ExpandDimsconv1d_35/Relu:activations:0(max_pooling1d_15/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_15/ExpandDims?
max_pooling1d_15/MaxPoolMaxPool$max_pooling1d_15/ExpandDims:output:0*0
_output_shapes
:??????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_15/MaxPool?
max_pooling1d_15/SqueezeSqueeze!max_pooling1d_15/MaxPool:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims
2
max_pooling1d_15/Squeeze?
conv1d_36/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_36/conv1d/ExpandDims/dim?
conv1d_36/conv1d/ExpandDims
ExpandDims!max_pooling1d_15/Squeeze:output:0(conv1d_36/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
conv1d_36/conv1d/ExpandDims?
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_36_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02.
,conv1d_36/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_36/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_36/conv1d/ExpandDims_1/dim?
conv1d_36/conv1d/ExpandDims_1
ExpandDims4conv1d_36/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_36/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d_36/conv1d/ExpandDims_1?
conv1d_36/conv1dConv2D$conv1d_36/conv1d/ExpandDims:output:0&conv1d_36/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_36/conv1d?
conv1d_36/conv1d/SqueezeSqueezeconv1d_36/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_36/conv1d/Squeeze?
 conv1d_36/BiasAdd/ReadVariableOpReadVariableOp)conv1d_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_36/BiasAdd/ReadVariableOp?
conv1d_36/BiasAddBiasAdd!conv1d_36/conv1d/Squeeze:output:0(conv1d_36/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_36/BiasAdd|
conv1d_36/ReluReluconv1d_36/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_36/Relu?
max_pooling1d_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_16/ExpandDims/dim?
max_pooling1d_16/ExpandDims
ExpandDimsconv1d_36/Relu:activations:0(max_pooling1d_16/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_16/ExpandDims?
max_pooling1d_16/MaxPoolMaxPool$max_pooling1d_16/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_16/MaxPool?
max_pooling1d_16/SqueezeSqueeze!max_pooling1d_16/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_16/Squeeze?
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp?
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_20/batchnorm/add/y?
$batch_normalization_20/batchnorm/addAddV27batch_normalization_20/batchnorm/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/add?
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/Rsqrt?
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOp?
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/mul?
&batch_normalization_20/batchnorm/mul_1Mul!max_pooling1d_16/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/mul_1?
1batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_1?
&batch_normalization_20/batchnorm/mul_2Mul9batch_normalization_20/batchnorm/ReadVariableOp_1:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_20/batchnorm/mul_2?
1batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_2?
$batch_normalization_20/batchnorm/subSub9batch_normalization_20/batchnorm/ReadVariableOp_2:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_20/batchnorm/sub?
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_20/batchnorm/add_1?
activation_20/ReluRelu*batch_normalization_20/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_20/Relu?
conv1d_37/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_37/conv1d/ExpandDims/dim?
conv1d_37/conv1d/ExpandDims
ExpandDims activation_20/Relu:activations:0(conv1d_37/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_37/conv1d/ExpandDims?
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_37/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_37/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_37/conv1d/ExpandDims_1/dim?
conv1d_37/conv1d/ExpandDims_1
ExpandDims4conv1d_37/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_37/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_37/conv1d/ExpandDims_1?
conv1d_37/conv1dConv2D$conv1d_37/conv1d/ExpandDims:output:0&conv1d_37/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv1d_37/conv1d?
conv1d_37/conv1d/SqueezeSqueezeconv1d_37/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_37/conv1d/Squeeze?
 conv1d_37/BiasAdd/ReadVariableOpReadVariableOp)conv1d_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_37/BiasAdd/ReadVariableOp?
conv1d_37/BiasAddBiasAdd!conv1d_37/conv1d/Squeeze:output:0(conv1d_37/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_37/BiasAdd?
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_37/conv1d_37/kernel/Regularizer/SquareSquareDconv1d_37/conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_37/conv1d_37/kernel/Regularizer/Square?
,conv1d_37/conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_37/conv1d_37/kernel/Regularizer/Const?
*conv1d_37/conv1d_37/kernel/Regularizer/SumSum1conv1d_37/conv1d_37/kernel/Regularizer/Square:y:05conv1d_37/conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/Sum?
,conv1d_37/conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_37/conv1d_37/kernel/Regularizer/mul/x?
*conv1d_37/conv1d_37/kernel/Regularizer/mulMul5conv1d_37/conv1d_37/kernel/Regularizer/mul/x:output:03conv1d_37/conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/mul?
,conv1d_37/conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_37/conv1d_37/kernel/Regularizer/add/x?
*conv1d_37/conv1d_37/kernel/Regularizer/addAddV25conv1d_37/conv1d_37/kernel/Regularizer/add/x:output:0.conv1d_37/conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_37/conv1d_37/kernel/Regularizer/add?
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_21/batchnorm/ReadVariableOp?
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_21/batchnorm/add/y?
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/add?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/Rsqrt?
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_21/batchnorm/mul/ReadVariableOp?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/mul?
&batch_normalization_21/batchnorm/mul_1Mulconv1d_37/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/mul_1?
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_1?
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_21/batchnorm/mul_2?
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_21/batchnorm/ReadVariableOp_2?
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_21/batchnorm/sub?
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_21/batchnorm/add_1?
activation_21/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_21/Relu?
conv1d_38/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_38/conv1d/ExpandDims/dim?
conv1d_38/conv1d/ExpandDims
ExpandDims activation_21/Relu:activations:0(conv1d_38/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_38/conv1d/ExpandDims?
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_38/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_38/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_38/conv1d/ExpandDims_1/dim?
conv1d_38/conv1d/ExpandDims_1
ExpandDims4conv1d_38/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_38/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_38/conv1d/ExpandDims_1?
conv1d_38/conv1dConv2D$conv1d_38/conv1d/ExpandDims:output:0&conv1d_38/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_38/conv1d?
conv1d_38/conv1d/SqueezeSqueezeconv1d_38/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_38/conv1d/Squeeze?
 conv1d_38/BiasAdd/ReadVariableOpReadVariableOp)conv1d_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_38/BiasAdd/ReadVariableOp?
conv1d_38/BiasAddBiasAdd!conv1d_38/conv1d/Squeeze:output:0(conv1d_38/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_38/BiasAdd?
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_38/conv1d_38/kernel/Regularizer/SquareSquareDconv1d_38/conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_38/conv1d_38/kernel/Regularizer/Square?
,conv1d_38/conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_38/conv1d_38/kernel/Regularizer/Const?
*conv1d_38/conv1d_38/kernel/Regularizer/SumSum1conv1d_38/conv1d_38/kernel/Regularizer/Square:y:05conv1d_38/conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/Sum?
,conv1d_38/conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_38/conv1d_38/kernel/Regularizer/mul/x?
*conv1d_38/conv1d_38/kernel/Regularizer/mulMul5conv1d_38/conv1d_38/kernel/Regularizer/mul/x:output:03conv1d_38/conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/mul?
,conv1d_38/conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_38/conv1d_38/kernel/Regularizer/add/x?
*conv1d_38/conv1d_38/kernel/Regularizer/addAddV25conv1d_38/conv1d_38/kernel/Regularizer/add/x:output:0.conv1d_38/conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_38/conv1d_38/kernel/Regularizer/add?

add_10/addAddV2conv1d_38/BiasAdd:output:0!max_pooling1d_16/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_10/add?
conv1d_39/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_39/conv1d/ExpandDims/dim?
conv1d_39/conv1d/ExpandDims
ExpandDimsadd_10/add:z:0(conv1d_39/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_39/conv1d/ExpandDims?
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_39_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_39/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_39/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_39/conv1d/ExpandDims_1/dim?
conv1d_39/conv1d/ExpandDims_1
ExpandDims4conv1d_39/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_39/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_39/conv1d/ExpandDims_1?
conv1d_39/conv1dConv2D$conv1d_39/conv1d/ExpandDims:output:0&conv1d_39/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_39/conv1d?
conv1d_39/conv1d/SqueezeSqueezeconv1d_39/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_39/conv1d/Squeeze?
 conv1d_39/BiasAdd/ReadVariableOpReadVariableOp)conv1d_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_39/BiasAdd/ReadVariableOp?
conv1d_39/BiasAddBiasAdd!conv1d_39/conv1d/Squeeze:output:0(conv1d_39/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_39/BiasAdd|
conv1d_39/ReluReluconv1d_39/BiasAdd:output:0*
T0*-
_output_shapes
:???????????2
conv1d_39/Relu?
max_pooling1d_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
max_pooling1d_17/ExpandDims/dim?
max_pooling1d_17/ExpandDims
ExpandDimsconv1d_39/Relu:activations:0(max_pooling1d_17/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
max_pooling1d_17/ExpandDims?
max_pooling1d_17/MaxPoolMaxPool$max_pooling1d_17/ExpandDims:output:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_17/MaxPool?
max_pooling1d_17/SqueezeSqueeze!max_pooling1d_17/MaxPool:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
max_pooling1d_17/Squeeze?
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_22/batchnorm/ReadVariableOp?
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_22/batchnorm/add/y?
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/add?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/Rsqrt?
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_22/batchnorm/mul/ReadVariableOp?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/mul?
&batch_normalization_22/batchnorm/mul_1Mul!max_pooling1d_17/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/mul_1?
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_22/batchnorm/ReadVariableOp_1?
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_22/batchnorm/mul_2?
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_22/batchnorm/ReadVariableOp_2?
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_22/batchnorm/sub?
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_22/batchnorm/add_1?
activation_22/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_22/Relu?
6conv1d_40/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?28
6conv1d_40/required_space_to_batch_paddings/input_shape?
8conv1d_40/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8conv1d_40/required_space_to_batch_paddings/base_paddings?
3conv1d_40/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        25
3conv1d_40/required_space_to_batch_paddings/paddings?
0conv1d_40/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        22
0conv1d_40/required_space_to_batch_paddings/crops?
$conv1d_40/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/SpaceToBatchND/block_shape?
!conv1d_40/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2#
!conv1d_40/SpaceToBatchND/paddings?
conv1d_40/SpaceToBatchNDSpaceToBatchND activation_22/Relu:activations:0-conv1d_40/SpaceToBatchND/block_shape:output:0*conv1d_40/SpaceToBatchND/paddings:output:0*
T0*,
_output_shapes
:?????????A?2
conv1d_40/SpaceToBatchND?
conv1d_40/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_40/conv1d/ExpandDims/dim?
conv1d_40/conv1d/ExpandDims
ExpandDims!conv1d_40/SpaceToBatchND:output:0(conv1d_40/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????A?2
conv1d_40/conv1d/ExpandDims?
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_40/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_40/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_40/conv1d/ExpandDims_1/dim?
conv1d_40/conv1d/ExpandDims_1
ExpandDims4conv1d_40/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_40/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_40/conv1d/ExpandDims_1?
conv1d_40/conv1dConv2D$conv1d_40/conv1d/ExpandDims:output:0&conv1d_40/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????A?*
paddingVALID*
strides
2
conv1d_40/conv1d?
conv1d_40/conv1d/SqueezeSqueezeconv1d_40/conv1d:output:0*
T0*,
_output_shapes
:?????????A?*
squeeze_dims
2
conv1d_40/conv1d/Squeeze?
$conv1d_40/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2&
$conv1d_40/BatchToSpaceND/block_shape?
conv1d_40/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2 
conv1d_40/BatchToSpaceND/crops?
conv1d_40/BatchToSpaceNDBatchToSpaceND!conv1d_40/conv1d/Squeeze:output:0-conv1d_40/BatchToSpaceND/block_shape:output:0'conv1d_40/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BatchToSpaceND?
 conv1d_40/BiasAdd/ReadVariableOpReadVariableOp)conv1d_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_40/BiasAdd/ReadVariableOp?
conv1d_40/BiasAddBiasAdd!conv1d_40/BatchToSpaceND:output:0(conv1d_40/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_40/BiasAdd?
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_40/conv1d_40/kernel/Regularizer/SquareSquareDconv1d_40/conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_40/conv1d_40/kernel/Regularizer/Square?
,conv1d_40/conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_40/conv1d_40/kernel/Regularizer/Const?
*conv1d_40/conv1d_40/kernel/Regularizer/SumSum1conv1d_40/conv1d_40/kernel/Regularizer/Square:y:05conv1d_40/conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/Sum?
,conv1d_40/conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_40/conv1d_40/kernel/Regularizer/mul/x?
*conv1d_40/conv1d_40/kernel/Regularizer/mulMul5conv1d_40/conv1d_40/kernel/Regularizer/mul/x:output:03conv1d_40/conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/mul?
,conv1d_40/conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_40/conv1d_40/kernel/Regularizer/add/x?
*conv1d_40/conv1d_40/kernel/Regularizer/addAddV25conv1d_40/conv1d_40/kernel/Regularizer/add/x:output:0.conv1d_40/conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_40/conv1d_40/kernel/Regularizer/add?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Mulconv1d_40/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/mul_1?
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_1?
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_2?
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2(
&batch_normalization_23/batchnorm/add_1?
activation_23/ReluRelu*batch_normalization_23/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2
activation_23/Relu?
conv1d_41/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_41/conv1d/ExpandDims/dim?
conv1d_41/conv1d/ExpandDims
ExpandDims activation_23/Relu:activations:0(conv1d_41/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????2
conv1d_41/conv1d/ExpandDims?
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02.
,conv1d_41/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_41/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_41/conv1d/ExpandDims_1/dim?
conv1d_41/conv1d/ExpandDims_1
ExpandDims4conv1d_41/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_41/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d_41/conv1d/ExpandDims_1?
conv1d_41/conv1dConv2D$conv1d_41/conv1d/ExpandDims:output:0&conv1d_41/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv1d_41/conv1d?
conv1d_41/conv1d/SqueezeSqueezeconv1d_41/conv1d:output:0*
T0*-
_output_shapes
:???????????*
squeeze_dims
2
conv1d_41/conv1d/Squeeze?
 conv1d_41/BiasAdd/ReadVariableOpReadVariableOp)conv1d_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv1d_41/BiasAdd/ReadVariableOp?
conv1d_41/BiasAddBiasAdd!conv1d_41/conv1d/Squeeze:output:0(conv1d_41/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
conv1d_41/BiasAdd?
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02>
<conv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
-conv1d_41/conv1d_41/kernel/Regularizer/SquareSquareDconv1d_41/conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2/
-conv1d_41/conv1d_41/kernel/Regularizer/Square?
,conv1d_41/conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,conv1d_41/conv1d_41/kernel/Regularizer/Const?
*conv1d_41/conv1d_41/kernel/Regularizer/SumSum1conv1d_41/conv1d_41/kernel/Regularizer/Square:y:05conv1d_41/conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/Sum?
,conv1d_41/conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2.
,conv1d_41/conv1d_41/kernel/Regularizer/mul/x?
*conv1d_41/conv1d_41/kernel/Regularizer/mulMul5conv1d_41/conv1d_41/kernel/Regularizer/mul/x:output:03conv1d_41/conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/mul?
,conv1d_41/conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,conv1d_41/conv1d_41/kernel/Regularizer/add/x?
*conv1d_41/conv1d_41/kernel/Regularizer/addAddV25conv1d_41/conv1d_41/kernel/Regularizer/add/x:output:0.conv1d_41/conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2,
*conv1d_41/conv1d_41/kernel/Regularizer/add?

add_11/addAddV2conv1d_41/BiasAdd:output:0!max_pooling1d_17/Squeeze:output:0*
T0*-
_output_shapes
:???????????2

add_11/add?
interoutput/IdentityIdentityadd_11/add:z:0*
T0*-
_output_shapes
:???????????2
interoutput/Identity[
reshape_16/ShapeShapeinput_6*
T0*
_output_shapes
:2
reshape_16/Shape?
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack?
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1?
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2?
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slice?
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2?
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shape?
reshape_16/ReshapeReshapeinput_6!reshape_16/Reshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
reshape_16/Reshape?
"tf_op_layer_concat_5/concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_5/concat_5/axis?
tf_op_layer_concat_5/concat_5ConcatV2interoutput/Identity:output:0reshape_16/Reshape:output:0+tf_op_layer_concat_5/concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2
tf_op_layer_concat_5/concat_5?
(second_dropout1/Tensordot/ReadVariableOpReadVariableOp1second_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(second_dropout1/Tensordot/ReadVariableOp?
second_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout1/Tensordot/axes?
second_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout1/Tensordot/free?
second_dropout1/Tensordot/ShapeShape&tf_op_layer_concat_5/concat_5:output:0*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/Shape?
'second_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/GatherV2/axis?
"second_dropout1/Tensordot/GatherV2GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/free:output:00second_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout1/Tensordot/GatherV2?
)second_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout1/Tensordot/GatherV2_1/axis?
$second_dropout1/Tensordot/GatherV2_1GatherV2(second_dropout1/Tensordot/Shape:output:0'second_dropout1/Tensordot/axes:output:02second_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout1/Tensordot/GatherV2_1?
second_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout1/Tensordot/Const?
second_dropout1/Tensordot/ProdProd+second_dropout1/Tensordot/GatherV2:output:0(second_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout1/Tensordot/Prod?
!second_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout1/Tensordot/Const_1?
 second_dropout1/Tensordot/Prod_1Prod-second_dropout1/Tensordot/GatherV2_1:output:0*second_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout1/Tensordot/Prod_1?
%second_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout1/Tensordot/concat/axis?
 second_dropout1/Tensordot/concatConcatV2'second_dropout1/Tensordot/free:output:0'second_dropout1/Tensordot/axes:output:0.second_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout1/Tensordot/concat?
second_dropout1/Tensordot/stackPack'second_dropout1/Tensordot/Prod:output:0)second_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout1/Tensordot/stack?
#second_dropout1/Tensordot/transpose	Transpose&tf_op_layer_concat_5/concat_5:output:0)second_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2%
#second_dropout1/Tensordot/transpose?
!second_dropout1/Tensordot/ReshapeReshape'second_dropout1/Tensordot/transpose:y:0(second_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout1/Tensordot/Reshape?
 second_dropout1/Tensordot/MatMulMatMul*second_dropout1/Tensordot/Reshape:output:00second_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 second_dropout1/Tensordot/MatMul?
!second_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2#
!second_dropout1/Tensordot/Const_2?
'second_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout1/Tensordot/concat_1/axis?
"second_dropout1/Tensordot/concat_1ConcatV2+second_dropout1/Tensordot/GatherV2:output:0*second_dropout1/Tensordot/Const_2:output:00second_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout1/Tensordot/concat_1?
second_dropout1/TensordotReshape*second_dropout1/Tensordot/MatMul:product:0+second_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Tensordot?
&second_dropout1/BiasAdd/ReadVariableOpReadVariableOp/second_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&second_dropout1/BiasAdd/ReadVariableOp?
second_dropout1/BiasAddBiasAdd"second_dropout1/Tensordot:output:0.second_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/BiasAdd?
second_dropout1/SigmoidSigmoid second_dropout1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
second_dropout1/Sigmoid?
%one_dropout1/Tensordot/ReadVariableOpReadVariableOp.one_dropout1_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%one_dropout1/Tensordot/ReadVariableOp?
one_dropout1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout1/Tensordot/axes?
one_dropout1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout1/Tensordot/free?
one_dropout1/Tensordot/ShapeShapeinteroutput/Identity:output:0*
T0*
_output_shapes
:2
one_dropout1/Tensordot/Shape?
$one_dropout1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/GatherV2/axis?
one_dropout1/Tensordot/GatherV2GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/free:output:0-one_dropout1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout1/Tensordot/GatherV2?
&one_dropout1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout1/Tensordot/GatherV2_1/axis?
!one_dropout1/Tensordot/GatherV2_1GatherV2%one_dropout1/Tensordot/Shape:output:0$one_dropout1/Tensordot/axes:output:0/one_dropout1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout1/Tensordot/GatherV2_1?
one_dropout1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout1/Tensordot/Const?
one_dropout1/Tensordot/ProdProd(one_dropout1/Tensordot/GatherV2:output:0%one_dropout1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod?
one_dropout1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout1/Tensordot/Const_1?
one_dropout1/Tensordot/Prod_1Prod*one_dropout1/Tensordot/GatherV2_1:output:0'one_dropout1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout1/Tensordot/Prod_1?
"one_dropout1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout1/Tensordot/concat/axis?
one_dropout1/Tensordot/concatConcatV2$one_dropout1/Tensordot/free:output:0$one_dropout1/Tensordot/axes:output:0+one_dropout1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/concat?
one_dropout1/Tensordot/stackPack$one_dropout1/Tensordot/Prod:output:0&one_dropout1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout1/Tensordot/stack?
 one_dropout1/Tensordot/transpose	Transposeinteroutput/Identity:output:0&one_dropout1/Tensordot/concat:output:0*
T0*-
_output_shapes
:???????????2"
 one_dropout1/Tensordot/transpose?
one_dropout1/Tensordot/ReshapeReshape$one_dropout1/Tensordot/transpose:y:0%one_dropout1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout1/Tensordot/Reshape?
one_dropout1/Tensordot/MatMulMatMul'one_dropout1/Tensordot/Reshape:output:0-one_dropout1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
one_dropout1/Tensordot/MatMul?
one_dropout1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2 
one_dropout1/Tensordot/Const_2?
$one_dropout1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout1/Tensordot/concat_1/axis?
one_dropout1/Tensordot/concat_1ConcatV2(one_dropout1/Tensordot/GatherV2:output:0'one_dropout1/Tensordot/Const_2:output:0-one_dropout1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout1/Tensordot/concat_1?
one_dropout1/TensordotReshape'one_dropout1/Tensordot/MatMul:product:0(one_dropout1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Tensordot?
#one_dropout1/BiasAdd/ReadVariableOpReadVariableOp,one_dropout1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#one_dropout1/BiasAdd/ReadVariableOp?
one_dropout1/BiasAddBiasAddone_dropout1/Tensordot:output:0+one_dropout1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/BiasAdd?
"one_dropout1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Max/reduction_indices?
one_dropout1/MaxMaxone_dropout1/BiasAdd:output:0+one_dropout1/Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Max?
one_dropout1/subSubone_dropout1/BiasAdd:output:0one_dropout1/Max:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/subx
one_dropout1/ExpExpone_dropout1/sub:z:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/Exp?
"one_dropout1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout1/Sum/reduction_indices?
one_dropout1/SumSumone_dropout1/Exp:y:0+one_dropout1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
one_dropout1/Sum?
one_dropout1/truedivRealDivone_dropout1/Exp:y:0one_dropout1/Sum:output:0*
T0*,
_output_shapes
:??????????@2
one_dropout1/truedivo
reshape_17/ShapeShapesecond_dropout1/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape_17/Shape?
reshape_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_17/strided_slice/stack?
 reshape_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_1?
 reshape_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_17/strided_slice/stack_2?
reshape_17/strided_sliceStridedSlicereshape_17/Shape:output:0'reshape_17/strided_slice/stack:output:0)reshape_17/strided_slice/stack_1:output:0)reshape_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_17/strided_slice?
reshape_17/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_17/Reshape/shape/1{
reshape_17/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_17/Reshape/shape/2?
reshape_17/Reshape/shapePack!reshape_17/strided_slice:output:0#reshape_17/Reshape/shape/1:output:0#reshape_17/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_17/Reshape/shape?
reshape_17/ReshapeReshapesecond_dropout1/Sigmoid:y:0!reshape_17/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_17/Reshapel
reshape_15/ShapeShapeone_dropout1/truediv:z:0*
T0*
_output_shapes
:2
reshape_15/Shape?
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack?
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1?
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2?
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slice?
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_15/Reshape/shape/1{
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
reshape_15/Reshape/shape/2?
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shape?
reshape_15/ReshapeReshapeone_dropout1/truediv:z:0!reshape_15/Reshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2
reshape_15/Reshape?
(second_dropout2/Tensordot/ReadVariableOpReadVariableOp1second_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02*
(second_dropout2/Tensordot/ReadVariableOp?
second_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
second_dropout2/Tensordot/axes?
second_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
second_dropout2/Tensordot/free?
second_dropout2/Tensordot/ShapeShapereshape_17/Reshape:output:0*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/Shape?
'second_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/GatherV2/axis?
"second_dropout2/Tensordot/GatherV2GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/free:output:00second_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"second_dropout2/Tensordot/GatherV2?
)second_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)second_dropout2/Tensordot/GatherV2_1/axis?
$second_dropout2/Tensordot/GatherV2_1GatherV2(second_dropout2/Tensordot/Shape:output:0'second_dropout2/Tensordot/axes:output:02second_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$second_dropout2/Tensordot/GatherV2_1?
second_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
second_dropout2/Tensordot/Const?
second_dropout2/Tensordot/ProdProd+second_dropout2/Tensordot/GatherV2:output:0(second_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
second_dropout2/Tensordot/Prod?
!second_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!second_dropout2/Tensordot/Const_1?
 second_dropout2/Tensordot/Prod_1Prod-second_dropout2/Tensordot/GatherV2_1:output:0*second_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 second_dropout2/Tensordot/Prod_1?
%second_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%second_dropout2/Tensordot/concat/axis?
 second_dropout2/Tensordot/concatConcatV2'second_dropout2/Tensordot/free:output:0'second_dropout2/Tensordot/axes:output:0.second_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 second_dropout2/Tensordot/concat?
second_dropout2/Tensordot/stackPack'second_dropout2/Tensordot/Prod:output:0)second_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
second_dropout2/Tensordot/stack?
#second_dropout2/Tensordot/transpose	Transposereshape_17/Reshape:output:0)second_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2%
#second_dropout2/Tensordot/transpose?
!second_dropout2/Tensordot/ReshapeReshape'second_dropout2/Tensordot/transpose:y:0(second_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!second_dropout2/Tensordot/Reshape?
 second_dropout2/Tensordot/MatMulMatMul*second_dropout2/Tensordot/Reshape:output:00second_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 second_dropout2/Tensordot/MatMul?
!second_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!second_dropout2/Tensordot/Const_2?
'second_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'second_dropout2/Tensordot/concat_1/axis?
"second_dropout2/Tensordot/concat_1ConcatV2+second_dropout2/Tensordot/GatherV2:output:0*second_dropout2/Tensordot/Const_2:output:00second_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"second_dropout2/Tensordot/concat_1?
second_dropout2/TensordotReshape*second_dropout2/Tensordot/MatMul:product:0+second_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Tensordot?
&second_dropout2/BiasAdd/ReadVariableOpReadVariableOp/second_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&second_dropout2/BiasAdd/ReadVariableOp?
second_dropout2/BiasAddBiasAdd"second_dropout2/Tensordot:output:0.second_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/BiasAdd?
second_dropout2/SigmoidSigmoid second_dropout2/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2
second_dropout2/Sigmoid?
%one_dropout2/Tensordot/ReadVariableOpReadVariableOp.one_dropout2_tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02'
%one_dropout2/Tensordot/ReadVariableOp?
one_dropout2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
one_dropout2/Tensordot/axes?
one_dropout2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
one_dropout2/Tensordot/free?
one_dropout2/Tensordot/ShapeShapereshape_15/Reshape:output:0*
T0*
_output_shapes
:2
one_dropout2/Tensordot/Shape?
$one_dropout2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/GatherV2/axis?
one_dropout2/Tensordot/GatherV2GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/free:output:0-one_dropout2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
one_dropout2/Tensordot/GatherV2?
&one_dropout2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&one_dropout2/Tensordot/GatherV2_1/axis?
!one_dropout2/Tensordot/GatherV2_1GatherV2%one_dropout2/Tensordot/Shape:output:0$one_dropout2/Tensordot/axes:output:0/one_dropout2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!one_dropout2/Tensordot/GatherV2_1?
one_dropout2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
one_dropout2/Tensordot/Const?
one_dropout2/Tensordot/ProdProd(one_dropout2/Tensordot/GatherV2:output:0%one_dropout2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod?
one_dropout2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
one_dropout2/Tensordot/Const_1?
one_dropout2/Tensordot/Prod_1Prod*one_dropout2/Tensordot/GatherV2_1:output:0'one_dropout2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
one_dropout2/Tensordot/Prod_1?
"one_dropout2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"one_dropout2/Tensordot/concat/axis?
one_dropout2/Tensordot/concatConcatV2$one_dropout2/Tensordot/free:output:0$one_dropout2/Tensordot/axes:output:0+one_dropout2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/concat?
one_dropout2/Tensordot/stackPack$one_dropout2/Tensordot/Prod:output:0&one_dropout2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
one_dropout2/Tensordot/stack?
 one_dropout2/Tensordot/transpose	Transposereshape_15/Reshape:output:0&one_dropout2/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2"
 one_dropout2/Tensordot/transpose?
one_dropout2/Tensordot/ReshapeReshape$one_dropout2/Tensordot/transpose:y:0%one_dropout2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2 
one_dropout2/Tensordot/Reshape?
one_dropout2/Tensordot/MatMulMatMul'one_dropout2/Tensordot/Reshape:output:0-one_dropout2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
one_dropout2/Tensordot/MatMul?
one_dropout2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2 
one_dropout2/Tensordot/Const_2?
$one_dropout2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$one_dropout2/Tensordot/concat_1/axis?
one_dropout2/Tensordot/concat_1ConcatV2(one_dropout2/Tensordot/GatherV2:output:0'one_dropout2/Tensordot/Const_2:output:0-one_dropout2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
one_dropout2/Tensordot/concat_1?
one_dropout2/TensordotReshape'one_dropout2/Tensordot/MatMul:product:0(one_dropout2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Tensordot?
#one_dropout2/BiasAdd/ReadVariableOpReadVariableOp,one_dropout2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#one_dropout2/BiasAdd/ReadVariableOp?
one_dropout2/BiasAddBiasAddone_dropout2/Tensordot:output:0+one_dropout2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/BiasAdd?
"one_dropout2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Max/reduction_indices?
one_dropout2/MaxMaxone_dropout2/BiasAdd:output:0+one_dropout2/Max/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Max?
one_dropout2/subSubone_dropout2/BiasAdd:output:0one_dropout2/Max:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/sub?
one_dropout2/ExpExpone_dropout2/sub:z:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/Exp?
"one_dropout2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"one_dropout2/Sum/reduction_indices?
one_dropout2/SumSumone_dropout2/Exp:y:0+one_dropout2/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
one_dropout2/Sum?
one_dropout2/truedivRealDivone_dropout2/Exp:y:0one_dropout2/Sum:output:0*
T0*4
_output_shapes"
 :??????????????????2
one_dropout2/truedivu
second_output/ShapeShapesecond_dropout2/Sigmoid:y:0*
T0*
_output_shapes
:2
second_output/Shape?
!second_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!second_output/strided_slice/stack?
#second_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_1?
#second_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#second_output/strided_slice/stack_2?
second_output/strided_sliceStridedSlicesecond_output/Shape:output:0*second_output/strided_slice/stack:output:0,second_output/strided_slice/stack_1:output:0,second_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
second_output/strided_slice?
second_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
second_output/Reshape/shape/1?
second_output/Reshape/shapePack$second_output/strided_slice:output:0&second_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
second_output/Reshape/shape?
second_output/ReshapeReshapesecond_dropout2/Sigmoid:y:0$second_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
second_output/Reshapel
one_output/ShapeShapeone_dropout2/truediv:z:0*
T0*
_output_shapes
:2
one_output/Shape?
one_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
one_output/strided_slice/stack?
 one_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_1?
 one_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 one_output/strided_slice/stack_2?
one_output/strided_sliceStridedSliceone_output/Shape:output:0'one_output/strided_slice/stack:output:0)one_output/strided_slice/stack_1:output:0)one_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
one_output/strided_slice?
one_output/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
one_output/Reshape/shape/1?
one_output/Reshape/shapePack!one_output/strided_slice:output:0#one_output/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
one_output/Reshape/shape?
one_output/ReshapeReshapeone_dropout2/truediv:z:0!one_output/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2
one_output/Reshape?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_37_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/add?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_38_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/add?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_40_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/add?
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv1d_41_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/addx
IdentityIdentityone_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity

Identity_1Identitysecond_output/Reshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
?
?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604174

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
9__inference_batch_normalization_23_layer_call_fn_49605076

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_conv1d_37_layer_call_fn_49600483

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?*
?
9__inference_batch_normalization_23_layer_call_fn_49605056

inputs
assignmovingavg_49605031
assignmovingavg_1_49605037)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49605031*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49605031*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605031*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605031*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49605031AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49605031*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49605037*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49605037*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605037*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605037*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49605037AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49605037*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
9__inference_batch_normalization_21_layer_call_fn_49604508

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
2__inference_second_dropout1_layer_call_fn_49605428

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAddf
SigmoidSigmoidBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2	
Sigmoidd
IdentityIdentitySigmoid:y:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????:::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
L
.__inference_interoutput_layer_call_fn_49605252

inputs

identity_1`
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????2

Identityo

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling1d_15_layer_call_fn_49600269

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?+
?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604266

inputs
assignmovingavg_49604241
assignmovingavg_1_49604247)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604241*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604241*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604241*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604241*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604241AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604241*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604247*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604247*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604247*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604247*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604247AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604247*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_conv1d_38_layer_call_fn_49600645

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_conv1d_35_layer_call_and_return_conditional_losses_49600234

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAdde
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????:::\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_conv1d_39_layer_call_fn_49600679

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?+
?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604544

inputs
assignmovingavg_49604519
assignmovingavg_1_49604525)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604519*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604519*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604519*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604519*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604519AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604519*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604525*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604525*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604525*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604525*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604525AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604525*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
J__inference_one_dropout1_layer_call_and_return_conditional_losses_49605329

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:???????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAddy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Max/reduction_indices?
MaxMaxBiasAdd:output:0Max/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
Maxh
subSubBiasAdd:output:0Max:output:0*
T0*,
_output_shapes
:??????????@2
subQ
ExpExpsub:z:0*
T0*,
_output_shapes
:??????????@2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indices?
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2
Sumk
truedivRealDivExp:y:0Sum:output:0*
T0*,
_output_shapes
:??????????@2	
truedivd
IdentityIdentitytruediv:z:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????:::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?+
?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604432

inputs
assignmovingavg_49604407
assignmovingavg_1_49604413)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604407*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604407*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604407*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604407*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604407AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604407*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604413*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604413*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604413*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604413*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604413AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604413*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_conv1d_36_layer_call_fn_49600303

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????@:::\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
j
N__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_49600312

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_reshape_15_layer_call_and_return_conditional_losses_49605441

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape}
ReshapeReshapeinputsReshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:???????????????????A2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?+
?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604722

inputs
assignmovingavg_49604697
assignmovingavg_1_49604703)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604697*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604697*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604697*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604697*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604697AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604697*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604703*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604703*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604703*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604703*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604703AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604703*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?*
?
9__inference_batch_normalization_22_layer_call_fn_49604890

inputs
assignmovingavg_49604865
assignmovingavg_1_49604871)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604865*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604865*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604865*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604865*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604865AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604865*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604871*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604871*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604871*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604871*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604871AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604871*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_reshape_17_layer_call_and_return_conditional_losses_49605467

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape}
ReshapeReshapeinputsReshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:???????????????????A2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
G__inference_conv1d_39_layer_call_and_return_conditional_losses_49600662

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relut
IdentityIdentityRelu:activations:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
2__inference_second_dropout2_layer_call_fn_49605616

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAddn
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2	
Sigmoidl
IdentityIdentitySigmoid:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????A:::] Y
5
_output_shapes#
!:???????????????????A
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
r
__inference_loss_fn_1_49605690?
;conv1d_38_kernel_regularizer_square_readvariableop_resource
identity??
2conv1d_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv1d_38_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_38/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_38/kernel/Regularizer/SquareSquare:conv1d_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_38/kernel/Regularizer/Square?
"conv1d_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_38/kernel/Regularizer/Const?
 conv1d_38/kernel/Regularizer/SumSum'conv1d_38/kernel/Regularizer/Square:y:0+conv1d_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/Sum?
"conv1d_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_38/kernel/Regularizer/mul/x?
 conv1d_38/kernel/Regularizer/mulMul+conv1d_38/kernel/Regularizer/mul/x:output:0)conv1d_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/mul?
"conv1d_38/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_38/kernel/Regularizer/add/x?
 conv1d_38/kernel/Regularizer/addAddV2+conv1d_38/kernel/Regularizer/add/x:output:0$conv1d_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_38/kernel/Regularizer/addg
IdentityIdentity$conv1d_38/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
j
N__inference_max_pooling1d_15_layer_call_and_return_conditional_losses_49600260

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?+
?
9__inference_batch_normalization_20_layer_call_fn_49604210

inputs
assignmovingavg_49604185
assignmovingavg_1_49604191)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604185*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604185*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604185*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604185*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604185AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604185*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604191*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604191*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604191*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604191*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604191AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604191*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604452

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
r
__inference_loss_fn_3_49605716?
;conv1d_41_kernel_regularizer_square_readvariableop_resource
identity??
2conv1d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv1d_41_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_41/kernel/Regularizer/SquareSquare:conv1d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_41/kernel/Regularizer/Square?
"conv1d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_41/kernel/Regularizer/Const?
 conv1d_41/kernel/Regularizer/SumSum'conv1d_41/kernel/Regularizer/Square:y:0+conv1d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/Sum?
"conv1d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_41/kernel/Regularizer/mul/x?
 conv1d_41/kernel/Regularizer/mulMul+conv1d_41/kernel/Regularizer/mul/x:output:0)conv1d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/mul?
"conv1d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_41/kernel/Regularizer/add/x?
 conv1d_41/kernel/Regularizer/addAddV2+conv1d_41/kernel/Regularizer/add/x:output:0$conv1d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_41/kernel/Regularizer/addg
IdentityIdentity$conv1d_41/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
j
N__inference_max_pooling1d_17_layer_call_and_return_conditional_losses_49600688

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_activation_22_layer_call_and_return_conditional_losses_49604915

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604742

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
9__inference_batch_normalization_23_layer_call_fn_49605188

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
p
D__inference_add_10_layer_call_and_return_conditional_losses_49604644
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:???????????2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:???????????:???????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
9__inference_batch_normalization_22_layer_call_fn_49604798

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?+
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605112

inputs
assignmovingavg_49605087
assignmovingavg_1_49605093)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49605087*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49605087*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605087*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605087*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49605087AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49605087*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49605093*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49605093*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605093*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605093*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49605093AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49605093*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?+
?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604834

inputs
assignmovingavg_49604809
assignmovingavg_1_49604815)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604809*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604809*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604809*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604809*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604809AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604809*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604815*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604815*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604815*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604815*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604815AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604815*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
M
.__inference_interoutput_layer_call_fn_49605247

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *O???2
dropout/Consty
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *33?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????2
dropout/Mul_1k
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
U
)__inference_add_10_layer_call_fn_49604650
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*-
_output_shapes
:???????????2
adda
IdentityIdentityadd:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:???????????:???????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?+
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605000

inputs
assignmovingavg_49604975
assignmovingavg_1_49604981)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604975*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604975*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604975*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604975*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604975AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604975*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604981*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604981*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604981*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604981*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604981AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604981*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?*
?
9__inference_batch_normalization_21_layer_call_fn_49604488

inputs
assignmovingavg_49604463
assignmovingavg_1_49604469)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604463*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604463*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604463*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604463*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604463AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604463*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604469*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604469*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604469*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604469*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604469AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604469*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?+
?
9__inference_batch_normalization_23_layer_call_fn_49605168

inputs
assignmovingavg_49605143
assignmovingavg_1_49605149)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49605143*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49605143*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605143*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49605143*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49605143AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49605143*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49605149*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49605149*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605149*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49605149*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49605149AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49605149*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_reshape_15_layer_call_fn_49605454

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?A2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape}
ReshapeReshapeinputsReshape/shape:output:0*
T0*5
_output_shapes#
!:???????????????????A2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:???????????????????A2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?+
?
9__inference_batch_normalization_22_layer_call_fn_49604778

inputs
assignmovingavg_49604753
assignmovingavg_1_49604759)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604753*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604753*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604753*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604753*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604753AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604753*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604759*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604759*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604759*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604759*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604759AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604759*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
g
K__inference_activation_23_layer_call_and_return_conditional_losses_49605193

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
M__inference_second_dropout2_layer_call_and_return_conditional_losses_49605585

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?A*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:???????????????????A2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2	
BiasAddn
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????2	
Sigmoidl
IdentityIdentitySigmoid:y:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????A:::] Y
5
_output_shapes#
!:???????????????????A
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_conv1d_37_layer_call_and_return_conditional_losses_49600459

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?p
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????????????:::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
9__inference_batch_normalization_21_layer_call_fn_49604620

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_reshape_16_layer_call_fn_49605278

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape|
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2	
Reshapeq
IdentityIdentityReshape:output:0*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
9__inference_batch_normalization_21_layer_call_fn_49604600

inputs
assignmovingavg_49604575
assignmovingavg_1_49604581)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604575*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604575*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604575*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604575*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604575AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604575*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604581*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604581*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604581*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604581*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604581AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604581*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604286

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_signature_wrapper_49604082
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37*3
Tin,
*2(*
Tout
2*L
_output_shapes:
8:??????????????????:??????????????????*H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_496002172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes?
?:??????????:::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_6:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
?
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605020

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1m
IdentityIdentitybatchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????:::::U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605132

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1u
IdentityIdentitybatchnorm/add_1:z:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????:::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_activation_21_layer_call_fn_49604630

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
7__inference_tf_op_layer_concat_5_layer_call_fn_49605292
inputs_0
inputs_1
identityi
concat_5/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_5/axis?
concat_5ConcatV2inputs_0inputs_1concat_5/axis:output:0*
N*
T0*
_cloned(*-
_output_shapes
:???????????2

concat_5k
IdentityIdentityconcat_5:output:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:???????????:??????????????????:W S
-
_output_shapes
:???????????
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/1
?(
?
,__inference_conv1d_40_layer_call_fn_49600881

inputs0
,required_space_to_batch_paddings_block_shape/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??
,required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:?2.
,required_space_to_batch_paddings/input_shape?
.required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        20
.required_space_to_batch_paddings/base_paddings?
)required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2+
)required_space_to_batch_paddings/paddings?
&required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2(
&required_space_to_batch_paddings/crops?
SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
SpaceToBatchND/block_shape?
SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2
SpaceToBatchND/paddings?
SpaceToBatchNDSpaceToBatchNDinputs#SpaceToBatchND/block_shape:output:0 SpaceToBatchND/paddings:output:0*
T0*5
_output_shapes#
!:???????????????????2
SpaceToBatchNDp
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsSpaceToBatchND:output:0conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
conv1d/Squeeze?
BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2
BatchToSpaceND/block_shape?
BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2
BatchToSpaceND/crops?
BatchToSpaceNDBatchToSpaceNDconv1d/Squeeze:output:0#BatchToSpaceND/block_shape:output:0BatchToSpaceND/crops:output:0*
T0*5
_output_shapes#
!:???????????????????2
BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddBatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAdd?
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/addr
IdentityIdentityBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:???????????????????::::] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_one_output_layer_call_and_return_conditional_losses_49605628

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
r
__inference_loss_fn_2_49605703?
;conv1d_40_kernel_regularizer_square_readvariableop_resource
identity??
2conv1d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv1d_40_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_40/kernel/Regularizer/SquareSquare:conv1d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_40/kernel/Regularizer/Square?
"conv1d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_40/kernel/Regularizer/Const?
 conv1d_40/kernel/Regularizer/SumSum'conv1d_40/kernel/Regularizer/Square:y:0+conv1d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/Sum?
"conv1d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_40/kernel/Regularizer/mul/x?
 conv1d_40/kernel/Regularizer/mulMul+conv1d_40/kernel/Regularizer/mul/x:output:0)conv1d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/mul?
"conv1d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_40/kernel/Regularizer/add/x?
 conv1d_40/kernel/Regularizer/addAddV2+conv1d_40/kernel/Regularizer/add/x:output:0$conv1d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_40/kernel/Regularizer/addg
IdentityIdentity$conv1d_40/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
O
3__inference_max_pooling1d_16_layer_call_fn_49600321

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_activation_22_layer_call_fn_49604920

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?*
?
9__inference_batch_normalization_20_layer_call_fn_49604322

inputs
assignmovingavg_49604297
assignmovingavg_1_49604303)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:???????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/49604297*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_49604297*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604297*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/49604297*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_49604297AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/49604297*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/49604303*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_49604303*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604303*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/49604303*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_49604303AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/49604303*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):???????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

L
0__inference_second_output_layer_call_fn_49605664

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
r
__inference_loss_fn_0_49605677?
;conv1d_37_kernel_regularizer_square_readvariableop_resource
identity??
2conv1d_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv1d_37_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:??*
dtype024
2conv1d_37/kernel/Regularizer/Square/ReadVariableOp?
#conv1d_37/kernel/Regularizer/SquareSquare:conv1d_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:??2%
#conv1d_37/kernel/Regularizer/Square?
"conv1d_37/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"conv1d_37/kernel/Regularizer/Const?
 conv1d_37/kernel/Regularizer/SumSum'conv1d_37/kernel/Regularizer/Square:y:0+conv1d_37/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/Sum?
"conv1d_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv1d_37/kernel/Regularizer/mul/x?
 conv1d_37/kernel/Regularizer/mulMul+conv1d_37/kernel/Regularizer/mul/x:output:0)conv1d_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/mul?
"conv1d_37/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv1d_37/kernel/Regularizer/add/x?
 conv1d_37/kernel/Regularizer/addAddV2+conv1d_37/kernel/Regularizer/add/x:output:0$conv1d_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv1d_37/kernel/Regularizer/addg
IdentityIdentity$conv1d_37/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
g
K__inference_activation_21_layer_call_and_return_conditional_losses_49604625

inputs
identityT
ReluReluinputs*
T0*-
_output_shapes
:???????????2
Relul
IdentityIdentityRelu:activations:0*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs"?.
saver_filename:0
Identity:0Identity_1268"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_65
serving_default_input_6:0??????????G

one_output9
StatefulPartitionedCall:0??????????????????J
second_output9
StatefulPartitionedCall:1??????????????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer_with_weights-10
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-11
layer-24
layer_with_weights-12
layer-25
layer-26
layer-27
layer_with_weights-13
layer-28
layer_with_weights-14
layer-29
layer-30
 layer-31
!	optimizer
"loss
#	variables
$regularization_losses
%trainable_variables
&	keras_api
'
signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_model??{"class_name": "Model", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1040, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_35", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_15", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_15", "inbound_nodes": [[["conv1d_35", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_36", "inbound_nodes": [[["max_pooling1d_15", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_16", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_16", "inbound_nodes": [[["conv1d_36", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["max_pooling1d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_37", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv1d_37", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_38", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "name": "add_10", "inbound_nodes": [[["conv1d_38", 0, 0, {}], ["max_pooling1d_16", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_39", "inbound_nodes": [[["add_10", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_17", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_17", "inbound_nodes": [[["conv1d_39", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["max_pooling1d_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_40", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv1d_40", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_41", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "name": "add_11", "inbound_nodes": [[["conv1d_41", 0, 0, {}], ["max_pooling1d_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "interoutput", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}, "name": "interoutput", "inbound_nodes": [[["add_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8]}}, "name": "reshape_16", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_5", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_5", "op": "ConcatV2", "input": ["interoutput_5/Identity", "reshape_16/Identity", "concat_5/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_5", "inbound_nodes": [[["interoutput", 0, 0, {}], ["reshape_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "one_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "one_dropout1", "inbound_nodes": [[["interoutput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "second_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "second_dropout1", "inbound_nodes": [[["tf_op_layer_concat_5", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}, "name": "reshape_15", "inbound_nodes": [[["one_dropout1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}, "name": "reshape_17", "inbound_nodes": [[["second_dropout1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "one_dropout2", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "one_dropout2", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "second_dropout2", "trainable": true, "dtype": "float32", "units": 7, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "second_dropout2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "one_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "one_output", "inbound_nodes": [[["one_dropout2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "second_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "second_output", "inbound_nodes": [[["second_dropout2", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": {"one_output": ["one_output", 0, 0], "second_output": ["second_output", 0, 0]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1040, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1040, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_35", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_15", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_15", "inbound_nodes": [[["conv1d_35", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_36", "inbound_nodes": [[["max_pooling1d_15", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_16", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_16", "inbound_nodes": [[["conv1d_36", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["max_pooling1d_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_37", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv1d_37", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_38", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "name": "add_10", "inbound_nodes": [[["conv1d_38", 0, 0, {}], ["max_pooling1d_16", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_39", "inbound_nodes": [[["add_10", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_17", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "max_pooling1d_17", "inbound_nodes": [[["conv1d_39", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["max_pooling1d_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_22", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_40", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv1d_40", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_41", "inbound_nodes": [[["activation_23", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "name": "add_11", "inbound_nodes": [[["conv1d_41", 0, 0, {}], ["max_pooling1d_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "interoutput", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}, "name": "interoutput", "inbound_nodes": [[["add_11", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8]}}, "name": "reshape_16", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_5", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_5", "op": "ConcatV2", "input": ["interoutput_5/Identity", "reshape_16/Identity", "concat_5/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_5", "inbound_nodes": [[["interoutput", 0, 0, {}], ["reshape_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "one_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "one_dropout1", "inbound_nodes": [[["interoutput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "second_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "second_dropout1", "inbound_nodes": [[["tf_op_layer_concat_5", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}, "name": "reshape_15", "inbound_nodes": [[["one_dropout1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}, "name": "reshape_17", "inbound_nodes": [[["second_dropout1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "one_dropout2", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "one_dropout2", "inbound_nodes": [[["reshape_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "second_dropout2", "trainable": true, "dtype": "float32", "units": 7, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "second_dropout2", "inbound_nodes": [[["reshape_17", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "one_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "one_output", "inbound_nodes": [[["one_dropout2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "second_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "second_output", "inbound_nodes": [[["second_dropout2", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": {"one_output": ["one_output", 0, 0], "second_output": ["second_output", 0, 0]}}}, "training_config": {"loss": {"one_output": "categorical_crossentropy", "second_output": {"class_name": "weight_sample_loss", "config": {"reduction": "auto", "name": "weight_sample_loss"}}}, "metrics": ["binary_accuracy", "f1_metric", "recall_metric", "precision_metric"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1040, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1040, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
?	

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_35", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1040, 1]}}
?
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling1d_15", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_36", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 520, 64]}}
?
8	variables
9regularization_losses
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling1d_16", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 260, 128]}}
?
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_37", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 260, 128]}}
?	
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 260, 128]}}
?
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


\kernel
]bias
^	variables
_regularization_losses
`trainable_variables
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_38", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 260, 128]}}
?
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_10", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 260, 128]}, {"class_name": "TensorShape", "items": [null, 260, 128]}]}
?	

fkernel
gbias
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_39", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 260, 128]}}
?
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling1d_17", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 256]}}
?
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


}kernel
~bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_40", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 256]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 256]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "relu"}}
?

?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1d_41", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 256]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Add", "name": "add_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "add_11", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 130, 256]}, {"class_name": "TensorShape", "items": [null, 130, 256]}]}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "interoutput", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "interoutput", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8]}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concat_5", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_5", "op": "ConcatV2", "input": ["interoutput_5/Identity", "reshape_16/Identity", "concat_5/axis"], "attr": {"T": {"type": "DT_FLOAT"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"2": -1}}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "one_dropout1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "one_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 256]}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "second_dropout1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "second_dropout1", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 264}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 130, 264]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_17", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 8320]}}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "one_dropout2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "one_dropout2", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 8320]}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "second_dropout2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "second_dropout2", "trainable": true, "dtype": "float32", "units": 7, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 8320]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "one_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "one_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "second_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "second_output", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate(m?)m?2m?3m?=m?>m?Im?Jm?Pm?Qm?\m?]m?fm?gm?qm?rm?}m?~m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?(v?)v?2v?3v?=v?>v?Iv?Jv?Pv?Qv?\v?]v?fv?gv?qv?rv?}v?~v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
(0
)1
22
33
=4
>5
?6
@7
I8
J9
P10
Q11
R12
S13
\14
]15
f16
g17
q18
r19
s20
t21
}22
~23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
(0
)1
22
33
=4
>5
I6
J7
P8
Q9
\10
]11
f12
g13
q14
r15
}16
~17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
#	variables
?layers
 ?layer_regularization_losses
?layer_metrics
?metrics
$regularization_losses
?non_trainable_variables
%trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$@2conv1d_35/kernel
:@2conv1d_35/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
?layers
*	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
+regularization_losses
?non_trainable_variables
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
.	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
/regularization_losses
?non_trainable_variables
0trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@?2conv1d_36/kernel
:?2conv1d_36/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?layers
4	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
5regularization_losses
?non_trainable_variables
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
8	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
9regularization_losses
?non_trainable_variables
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_20/gamma
*:(?2batch_normalization_20/beta
3:1? (2"batch_normalization_20/moving_mean
7:5? (2&batch_normalization_20/moving_variance
<
=0
>1
?2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?layers
A	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Bregularization_losses
?non_trainable_variables
Ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
E	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Fregularization_losses
?non_trainable_variables
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv1d_37/kernel
:?2conv1d_37/bias
.
I0
J1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?layers
K	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Lregularization_losses
?non_trainable_variables
Mtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_21/gamma
*:(?2batch_normalization_21/beta
3:1? (2"batch_normalization_21/moving_mean
7:5? (2&batch_normalization_21/moving_variance
<
P0
Q1
R2
S3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?layers
T	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Uregularization_losses
?non_trainable_variables
Vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
X	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
Yregularization_losses
?non_trainable_variables
Ztrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv1d_38/kernel
:?2conv1d_38/bias
.
\0
]1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
?
?layers
^	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
_regularization_losses
?non_trainable_variables
`trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
b	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
cregularization_losses
?non_trainable_variables
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv1d_39/kernel
:?2conv1d_39/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
?layers
h	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
iregularization_losses
?non_trainable_variables
jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
l	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
mregularization_losses
?non_trainable_variables
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_22/gamma
*:(?2batch_normalization_22/beta
3:1? (2"batch_normalization_22/moving_mean
7:5? (2&batch_normalization_22/moving_variance
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?layers
u	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
vregularization_losses
?non_trainable_variables
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
y	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
zregularization_losses
?non_trainable_variables
{trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv1d_40/kernel
:?2conv1d_40/bias
.
}0
~1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
?
?layers
	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_23/gamma
*:(?2batch_normalization_23/beta
3:1? (2"batch_normalization_23/moving_mean
7:5? (2&batch_normalization_23/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv1d_41/kernel
:?2conv1d_41/bias
0
?0
?1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	?@2one_dropout1_5/kernel
!:@2one_dropout1_5/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)	?@2second_dropout1_5/kernel
$:"@2second_dropout1_5/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	?A2one_dropout2_5/kernel
!:2one_dropout2_5/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)	?A2second_dropout2_5/kernel
$:"2second_dropout2_5/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?	variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
y
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10"
trackable_list_wrapper
Z
?0
@1
R2
S3
s4
t5
?6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "one_output_loss", "dtype": "float32", "config": {"name": "one_output_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "second_output_loss", "dtype": "float32", "config": {"name": "second_output_loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "one_output_binary_accuracy", "dtype": "float32", "config": {"name": "one_output_binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "one_output_f1_metric", "dtype": "float32", "config": {"name": "one_output_f1_metric", "dtype": "float32", "fn": "f1_metric"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "one_output_recall_metric", "dtype": "float32", "config": {"name": "one_output_recall_metric", "dtype": "float32", "fn": "recall_metric"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "one_output_precision_metric", "dtype": "float32", "config": {"name": "one_output_precision_metric", "dtype": "float32", "fn": "precision_metric"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "second_output_binary_accuracy", "dtype": "float32", "config": {"name": "second_output_binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "second_output_f1_metric", "dtype": "float32", "config": {"name": "second_output_f1_metric", "dtype": "float32", "fn": "f1_metric"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "second_output_recall_metric", "dtype": "float32", "config": {"name": "second_output_recall_metric", "dtype": "float32", "fn": "recall_metric"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "second_output_precision_metric", "dtype": "float32", "config": {"name": "second_output_precision_metric", "dtype": "float32", "fn": "precision_metric"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
+:)@2Adam/conv1d_35/kernel/m
!:@2Adam/conv1d_35/bias/m
,:*@?2Adam/conv1d_36/kernel/m
": ?2Adam/conv1d_36/bias/m
0:.?2#Adam/batch_normalization_20/gamma/m
/:-?2"Adam/batch_normalization_20/beta/m
-:+??2Adam/conv1d_37/kernel/m
": ?2Adam/conv1d_37/bias/m
0:.?2#Adam/batch_normalization_21/gamma/m
/:-?2"Adam/batch_normalization_21/beta/m
-:+??2Adam/conv1d_38/kernel/m
": ?2Adam/conv1d_38/bias/m
-:+??2Adam/conv1d_39/kernel/m
": ?2Adam/conv1d_39/bias/m
0:.?2#Adam/batch_normalization_22/gamma/m
/:-?2"Adam/batch_normalization_22/beta/m
-:+??2Adam/conv1d_40/kernel/m
": ?2Adam/conv1d_40/bias/m
0:.?2#Adam/batch_normalization_23/gamma/m
/:-?2"Adam/batch_normalization_23/beta/m
-:+??2Adam/conv1d_41/kernel/m
": ?2Adam/conv1d_41/bias/m
-:+	?@2Adam/one_dropout1_5/kernel/m
&:$@2Adam/one_dropout1_5/bias/m
0:.	?@2Adam/second_dropout1_5/kernel/m
):'@2Adam/second_dropout1_5/bias/m
-:+	?A2Adam/one_dropout2_5/kernel/m
&:$2Adam/one_dropout2_5/bias/m
0:.	?A2Adam/second_dropout2_5/kernel/m
):'2Adam/second_dropout2_5/bias/m
+:)@2Adam/conv1d_35/kernel/v
!:@2Adam/conv1d_35/bias/v
,:*@?2Adam/conv1d_36/kernel/v
": ?2Adam/conv1d_36/bias/v
0:.?2#Adam/batch_normalization_20/gamma/v
/:-?2"Adam/batch_normalization_20/beta/v
-:+??2Adam/conv1d_37/kernel/v
": ?2Adam/conv1d_37/bias/v
0:.?2#Adam/batch_normalization_21/gamma/v
/:-?2"Adam/batch_normalization_21/beta/v
-:+??2Adam/conv1d_38/kernel/v
": ?2Adam/conv1d_38/bias/v
-:+??2Adam/conv1d_39/kernel/v
": ?2Adam/conv1d_39/bias/v
0:.?2#Adam/batch_normalization_22/gamma/v
/:-?2"Adam/batch_normalization_22/beta/v
-:+??2Adam/conv1d_40/kernel/v
": ?2Adam/conv1d_40/bias/v
0:.?2#Adam/batch_normalization_23/gamma/v
/:-?2"Adam/batch_normalization_23/beta/v
-:+??2Adam/conv1d_41/kernel/v
": ?2Adam/conv1d_41/bias/v
-:+	?@2Adam/one_dropout1_5/kernel/v
&:$@2Adam/one_dropout1_5/bias/v
0:.	?@2Adam/second_dropout1_5/kernel/v
):'@2Adam/second_dropout1_5/bias/v
-:+	?A2Adam/one_dropout2_5/kernel/v
&:$2Adam/one_dropout2_5/bias/v
0:.	?A2Adam/second_dropout2_5/kernel/v
):'2Adam/second_dropout2_5/bias/v
?2?
#__inference__wrapped_model_49600217?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_6??????????
?2?
E__inference_model_5_layer_call_and_return_conditional_losses_49602515
E__inference_model_5_layer_call_and_return_conditional_losses_49602923?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_5_layer_call_fn_49603811
*__inference_model_5_layer_call_fn_49603403?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_conv1d_35_layer_call_and_return_conditional_losses_49600234?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
,__inference_conv1d_35_layer_call_fn_49600251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
N__inference_max_pooling1d_15_layer_call_and_return_conditional_losses_49600260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
3__inference_max_pooling1d_15_layer_call_fn_49600269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
G__inference_conv1d_36_layer_call_and_return_conditional_losses_49600286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????@
?2?
,__inference_conv1d_36_layer_call_fn_49600303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????@
?2?
N__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_49600312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
3__inference_max_pooling1d_16_layer_call_fn_49600321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604174
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604154
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604266
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604286?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_20_layer_call_fn_49604230
9__inference_batch_normalization_20_layer_call_fn_49604342
9__inference_batch_normalization_20_layer_call_fn_49604210
9__inference_batch_normalization_20_layer_call_fn_49604322?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_activation_20_layer_call_and_return_conditional_losses_49604347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_activation_20_layer_call_fn_49604352?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv1d_37_layer_call_and_return_conditional_losses_49600459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
,__inference_conv1d_37_layer_call_fn_49600483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604564
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604544
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604452
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604432?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_21_layer_call_fn_49604620
9__inference_batch_normalization_21_layer_call_fn_49604600
9__inference_batch_normalization_21_layer_call_fn_49604488
9__inference_batch_normalization_21_layer_call_fn_49604508?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_activation_21_layer_call_and_return_conditional_losses_49604625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_activation_21_layer_call_fn_49604630?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv1d_38_layer_call_and_return_conditional_losses_49600621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
,__inference_conv1d_38_layer_call_fn_49600645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
D__inference_add_10_layer_call_and_return_conditional_losses_49604644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_add_10_layer_call_fn_49604650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv1d_39_layer_call_and_return_conditional_losses_49600662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
,__inference_conv1d_39_layer_call_fn_49600679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
N__inference_max_pooling1d_17_layer_call_and_return_conditional_losses_49600688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
3__inference_max_pooling1d_17_layer_call_fn_49600697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604722
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604742
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604854
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604834?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_22_layer_call_fn_49604890
9__inference_batch_normalization_22_layer_call_fn_49604798
9__inference_batch_normalization_22_layer_call_fn_49604778
9__inference_batch_normalization_22_layer_call_fn_49604910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_activation_22_layer_call_and_return_conditional_losses_49604915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_activation_22_layer_call_fn_49604920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv1d_40_layer_call_and_return_conditional_losses_49600846?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
,__inference_conv1d_40_layer_call_fn_49600881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605112
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605132
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605000
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605020?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_23_layer_call_fn_49605076
9__inference_batch_normalization_23_layer_call_fn_49605188
9__inference_batch_normalization_23_layer_call_fn_49605168
9__inference_batch_normalization_23_layer_call_fn_49605056?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_activation_23_layer_call_and_return_conditional_losses_49605193?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_activation_23_layer_call_fn_49605198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv1d_41_layer_call_and_return_conditional_losses_49601019?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
,__inference_conv1d_41_layer_call_fn_49601043?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#???????????????????
?2?
D__inference_add_11_layer_call_and_return_conditional_losses_49605212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_add_11_layer_call_fn_49605218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_interoutput_layer_call_and_return_conditional_losses_49605235
I__inference_interoutput_layer_call_and_return_conditional_losses_49605230?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_interoutput_layer_call_fn_49605252
.__inference_interoutput_layer_call_fn_49605247?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_reshape_16_layer_call_and_return_conditional_losses_49605265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_reshape_16_layer_call_fn_49605278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_tf_op_layer_concat_5_layer_call_and_return_conditional_losses_49605285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_tf_op_layer_concat_5_layer_call_fn_49605292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_one_dropout1_layer_call_and_return_conditional_losses_49605329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_one_dropout1_layer_call_fn_49605366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_second_dropout1_layer_call_and_return_conditional_losses_49605397?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_second_dropout1_layer_call_fn_49605428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_reshape_15_layer_call_and_return_conditional_losses_49605441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_reshape_15_layer_call_fn_49605454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_reshape_17_layer_call_and_return_conditional_losses_49605467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_reshape_17_layer_call_fn_49605480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_one_dropout2_layer_call_and_return_conditional_losses_49605517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_one_dropout2_layer_call_fn_49605554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_second_dropout2_layer_call_and_return_conditional_losses_49605585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_second_dropout2_layer_call_fn_49605616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_one_output_layer_call_and_return_conditional_losses_49605628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_one_output_layer_call_fn_49605640?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_second_output_layer_call_and_return_conditional_losses_49605652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_second_output_layer_call_fn_49605664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_49605677?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_49605690?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_49605703?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_49605716?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
5B3
&__inference_signature_wrapper_49604082input_6
	J
Const?
#__inference__wrapped_model_49600217?6()23@=?>IJSPRQ\]fgtqsr?}~??????????????5?2
+?(
&?#
input_6??????????
? "???
;

one_output-?*

one_output??????????????????
A
second_output0?-
second_output???????????????????
K__inference_activation_20_layer_call_and_return_conditional_losses_49604347d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_activation_20_layer_call_fn_49604352W5?2
+?(
&?#
inputs???????????
? "?????????????
K__inference_activation_21_layer_call_and_return_conditional_losses_49604625d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_activation_21_layer_call_fn_49604630W5?2
+?(
&?#
inputs???????????
? "?????????????
K__inference_activation_22_layer_call_and_return_conditional_losses_49604915d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_activation_22_layer_call_fn_49604920W5?2
+?(
&?#
inputs???????????
? "?????????????
K__inference_activation_23_layer_call_and_return_conditional_losses_49605193d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
0__inference_activation_23_layer_call_fn_49605198W5?2
+?(
&?#
inputs???????????
? "?????????????
D__inference_add_10_layer_call_and_return_conditional_losses_49604644?f?c
\?Y
W?T
(?%
inputs/0???????????
(?%
inputs/1???????????
? "+?(
!?
0???????????
? ?
)__inference_add_10_layer_call_fn_49604650?f?c
\?Y
W?T
(?%
inputs/0???????????
(?%
inputs/1???????????
? "?????????????
D__inference_add_11_layer_call_and_return_conditional_losses_49605212?f?c
\?Y
W?T
(?%
inputs/0???????????
(?%
inputs/1???????????
? "+?(
!?
0???????????
? ?
)__inference_add_11_layer_call_fn_49605218?f?c
\?Y
W?T
(?%
inputs/0???????????
(?%
inputs/1???????????
? "?????????????
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604154~?@=>A?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604174~@=?>A?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604266n?@=>9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_49604286n@=?>9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
9__inference_batch_normalization_20_layer_call_fn_49604210q?@=>A?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
9__inference_batch_normalization_20_layer_call_fn_49604230q@=?>A?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
9__inference_batch_normalization_20_layer_call_fn_49604322a?@=>9?6
/?,
&?#
inputs???????????
p
? "?????????????
9__inference_batch_normalization_20_layer_call_fn_49604342a@=?>9?6
/?,
&?#
inputs???????????
p 
? "?????????????
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604432nRSPQ9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604452nSPRQ9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604544~RSPQA?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_49604564~SPRQA?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
9__inference_batch_normalization_21_layer_call_fn_49604488aRSPQ9?6
/?,
&?#
inputs???????????
p
? "?????????????
9__inference_batch_normalization_21_layer_call_fn_49604508aSPRQ9?6
/?,
&?#
inputs???????????
p 
? "?????????????
9__inference_batch_normalization_21_layer_call_fn_49604600qRSPQA?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
9__inference_batch_normalization_21_layer_call_fn_49604620qSPRQA?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604722~stqrA?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604742~tqsrA?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604834nstqr9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_49604854ntqsr9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
9__inference_batch_normalization_22_layer_call_fn_49604778qstqrA?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
9__inference_batch_normalization_22_layer_call_fn_49604798qtqsrA?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
9__inference_batch_normalization_22_layer_call_fn_49604890astqr9?6
/?,
&?#
inputs???????????
p
? "?????????????
9__inference_batch_normalization_22_layer_call_fn_49604910atqsr9?6
/?,
&?#
inputs???????????
p 
? "?????????????
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605000r????9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605020r????9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605112?????A?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_49605132?????A?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
9__inference_batch_normalization_23_layer_call_fn_49605056e????9?6
/?,
&?#
inputs???????????
p
? "?????????????
9__inference_batch_normalization_23_layer_call_fn_49605076e????9?6
/?,
&?#
inputs???????????
p 
? "?????????????
9__inference_batch_normalization_23_layer_call_fn_49605168u????A?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
9__inference_batch_normalization_23_layer_call_fn_49605188u????A?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
G__inference_conv1d_35_layer_call_and_return_conditional_losses_49600234v()<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????@
? ?
,__inference_conv1d_35_layer_call_fn_49600251i()<?9
2?/
-?*
inputs??????????????????
? "%?"??????????????????@?
G__inference_conv1d_36_layer_call_and_return_conditional_losses_49600286w23<?9
2?/
-?*
inputs??????????????????@
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_36_layer_call_fn_49600303j23<?9
2?/
-?*
inputs??????????????????@
? "&?#????????????????????
G__inference_conv1d_37_layer_call_and_return_conditional_losses_49600459xIJ=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_37_layer_call_fn_49600483kIJ=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
G__inference_conv1d_38_layer_call_and_return_conditional_losses_49600621x\]=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_38_layer_call_fn_49600645k\]=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
G__inference_conv1d_39_layer_call_and_return_conditional_losses_49600662xfg=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_39_layer_call_fn_49600679kfg=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
G__inference_conv1d_40_layer_call_and_return_conditional_losses_49600846z?}~=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_40_layer_call_fn_49600881m?}~=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
G__inference_conv1d_41_layer_call_and_return_conditional_losses_49601019z??=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
,__inference_conv1d_41_layer_call_fn_49601043m??=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
I__inference_interoutput_layer_call_and_return_conditional_losses_49605230h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
I__inference_interoutput_layer_call_and_return_conditional_losses_49605235h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
.__inference_interoutput_layer_call_fn_49605247[9?6
/?,
&?#
inputs???????????
p
? "?????????????
.__inference_interoutput_layer_call_fn_49605252[9?6
/?,
&?#
inputs???????????
p 
? "????????????=
__inference_loss_fn_0_49605677I?

? 
? "? =
__inference_loss_fn_1_49605690\?

? 
? "? =
__inference_loss_fn_2_49605703}?

? 
? "? >
__inference_loss_fn_3_49605716??

? 
? "? ?
N__inference_max_pooling1d_15_layer_call_and_return_conditional_losses_49600260?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
3__inference_max_pooling1d_15_layer_call_fn_49600269wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
N__inference_max_pooling1d_16_layer_call_and_return_conditional_losses_49600312?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
3__inference_max_pooling1d_16_layer_call_fn_49600321wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
N__inference_max_pooling1d_17_layer_call_and_return_conditional_losses_49600688?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
3__inference_max_pooling1d_17_layer_call_fn_49600697wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
E__inference_model_5_layer_call_and_return_conditional_losses_49602515?6()23?@=>IJRSPQ\]fgstqr?}~??????????????=?:
3?0
&?#
input_6??????????
p

 
? "???
???
=

one_output/?,
0/one_output??????????????????
C
second_output2?/
0/second_output??????????????????
? ?
E__inference_model_5_layer_call_and_return_conditional_losses_49602923?6()23@=?>IJSPRQ\]fgtqsr?}~??????????????=?:
3?0
&?#
input_6??????????
p 

 
? "???
???
=

one_output/?,
0/one_output??????????????????
C
second_output2?/
0/second_output??????????????????
? ?
*__inference_model_5_layer_call_fn_49603403?6()23?@=>IJRSPQ\]fgstqr?}~??????????????=?:
3?0
&?#
input_6??????????
p

 
? "???
;

one_output-?*

one_output??????????????????
A
second_output0?-
second_output???????????????????
*__inference_model_5_layer_call_fn_49603811?6()23@=?>IJSPRQ\]fgtqsr?}~??????????????=?:
3?0
&?#
input_6??????????
p 

 
? "???
;

one_output-?*

one_output??????????????????
A
second_output0?-
second_output???????????????????
J__inference_one_dropout1_layer_call_and_return_conditional_losses_49605329i??5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????@
? ?
/__inference_one_dropout1_layer_call_fn_49605366\??5?2
+?(
&?#
inputs???????????
? "???????????@?
J__inference_one_dropout2_layer_call_and_return_conditional_losses_49605517y??=?:
3?0
.?+
inputs???????????????????A
? "2?/
(?%
0??????????????????
? ?
/__inference_one_dropout2_layer_call_fn_49605554l??=?:
3?0
.?+
inputs???????????????????A
? "%?"???????????????????
H__inference_one_output_layer_call_and_return_conditional_losses_49605628n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
-__inference_one_output_layer_call_fn_49605640a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
H__inference_reshape_15_layer_call_and_return_conditional_losses_49605441k4?1
*?'
%?"
inputs??????????@
? "3?0
)?&
0???????????????????A
? ?
-__inference_reshape_15_layer_call_fn_49605454^4?1
*?'
%?"
inputs??????????@
? "&?#???????????????????A?
H__inference_reshape_16_layer_call_and_return_conditional_losses_49605265j4?1
*?'
%?"
inputs??????????
? "2?/
(?%
0??????????????????
? ?
-__inference_reshape_16_layer_call_fn_49605278]4?1
*?'
%?"
inputs??????????
? "%?"???????????????????
H__inference_reshape_17_layer_call_and_return_conditional_losses_49605467k4?1
*?'
%?"
inputs??????????@
? "3?0
)?&
0???????????????????A
? ?
-__inference_reshape_17_layer_call_fn_49605480^4?1
*?'
%?"
inputs??????????@
? "&?#???????????????????A?
M__inference_second_dropout1_layer_call_and_return_conditional_losses_49605397i??5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????@
? ?
2__inference_second_dropout1_layer_call_fn_49605428\??5?2
+?(
&?#
inputs???????????
? "???????????@?
M__inference_second_dropout2_layer_call_and_return_conditional_losses_49605585y??=?:
3?0
.?+
inputs???????????????????A
? "2?/
(?%
0??????????????????
? ?
2__inference_second_dropout2_layer_call_fn_49605616l??=?:
3?0
.?+
inputs???????????????????A
? "%?"???????????????????
K__inference_second_output_layer_call_and_return_conditional_losses_49605652n<?9
2?/
-?*
inputs??????????????????
? ".?+
$?!
0??????????????????
? ?
0__inference_second_output_layer_call_fn_49605664a<?9
2?/
-?*
inputs??????????????????
? "!????????????????????
&__inference_signature_wrapper_49604082?6()23@=?>IJSPRQ\]fgtqsr?}~??????????????@?=
? 
6?3
1
input_6&?#
input_6??????????"???
;

one_output-?*

one_output??????????????????
A
second_output0?-
second_output???????????????????
R__inference_tf_op_layer_concat_5_layer_call_and_return_conditional_losses_49605285?m?j
c?`
^?[
(?%
inputs/0???????????
/?,
inputs/1??????????????????
? "+?(
!?
0???????????
? ?
7__inference_tf_op_layer_concat_5_layer_call_fn_49605292?m?j
c?`
^?[
(?%
inputs/0???????????
/?,
inputs/1??????????????????
? "????????????