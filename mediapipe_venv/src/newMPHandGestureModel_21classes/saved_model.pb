??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
newDenseLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@*&
shared_namenewDenseLayer1/kernel

)newDenseLayer1/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer1/kernel*
_output_shapes

:*@*
dtype0
~
newDenseLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namenewDenseLayer1/bias
w
'newDenseLayer1/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer1/bias*
_output_shapes
:@*
dtype0
?
newDenseLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*&
shared_namenewDenseLayer2/kernel
?
)newDenseLayer2/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer2/kernel*
_output_shapes
:	@?*
dtype0

newDenseLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namenewDenseLayer2/bias
x
'newDenseLayer2/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer2/bias*
_output_shapes	
:?*
dtype0
?
newDenseLayer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_namenewDenseLayer3/kernel
?
)newDenseLayer3/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer3/kernel* 
_output_shapes
:
??*
dtype0

newDenseLayer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namenewDenseLayer3/bias
x
'newDenseLayer3/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer3/bias*
_output_shapes	
:?*
dtype0
?
newDenseLayer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_namenewDenseLayer4/kernel
?
)newDenseLayer4/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer4/kernel* 
_output_shapes
:
??*
dtype0

newDenseLayer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namenewDenseLayer4/bias
x
'newDenseLayer4/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer4/bias*
_output_shapes	
:?*
dtype0
?
newDenseLayer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_namenewDenseLayer5/kernel
?
)newDenseLayer5/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer5/kernel* 
_output_shapes
:
??*
dtype0

newDenseLayer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namenewDenseLayer5/bias
x
'newDenseLayer5/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer5/bias*
_output_shapes	
:?*
dtype0
?
newDenseLayer6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_namenewDenseLayer6/kernel
?
)newDenseLayer6/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer6/kernel* 
_output_shapes
:
??*
dtype0

newDenseLayer6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namenewDenseLayer6/bias
x
'newDenseLayer6/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer6/bias*
_output_shapes	
:?*
dtype0
?
newDenseLayer7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_namenewDenseLayer7/kernel
?
)newDenseLayer7/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayer7/kernel*
_output_shapes
:	?@*
dtype0
~
newDenseLayer7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namenewDenseLayer7/bias
w
'newDenseLayer7/bias/Read/ReadVariableOpReadVariableOpnewDenseLayer7/bias*
_output_shapes
:@*
dtype0
?
newDenseLayerOutput/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namenewDenseLayerOutput/kernel
?
.newDenseLayerOutput/kernel/Read/ReadVariableOpReadVariableOpnewDenseLayerOutput/kernel*
_output_shapes

:@*
dtype0
?
newDenseLayerOutput/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namenewDenseLayerOutput/bias
?
,newDenseLayerOutput/bias/Read/ReadVariableOpReadVariableOpnewDenseLayerOutput/bias*
_output_shapes
:*
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
?
Adam/newDenseLayer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@*-
shared_nameAdam/newDenseLayer1/kernel/m
?
0Adam/newDenseLayer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer1/kernel/m*
_output_shapes

:*@*
dtype0
?
Adam/newDenseLayer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/newDenseLayer1/bias/m
?
.Adam/newDenseLayer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/newDenseLayer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*-
shared_nameAdam/newDenseLayer2/kernel/m
?
0Adam/newDenseLayer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer2/kernel/m*
_output_shapes
:	@?*
dtype0
?
Adam/newDenseLayer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer2/bias/m
?
.Adam/newDenseLayer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer3/kernel/m
?
0Adam/newDenseLayer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer3/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer3/bias/m
?
.Adam/newDenseLayer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer4/kernel/m
?
0Adam/newDenseLayer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer4/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer4/bias/m
?
.Adam/newDenseLayer4/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer5/kernel/m
?
0Adam/newDenseLayer5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer5/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer5/bias/m
?
.Adam/newDenseLayer5/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer6/kernel/m
?
0Adam/newDenseLayer6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer6/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer6/bias/m
?
.Adam/newDenseLayer6/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer6/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_nameAdam/newDenseLayer7/kernel/m
?
0Adam/newDenseLayer7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer7/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/newDenseLayer7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/newDenseLayer7/bias/m
?
.Adam/newDenseLayer7/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer7/bias/m*
_output_shapes
:@*
dtype0
?
!Adam/newDenseLayerOutput/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/newDenseLayerOutput/kernel/m
?
5Adam/newDenseLayerOutput/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/newDenseLayerOutput/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/newDenseLayerOutput/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/newDenseLayerOutput/bias/m
?
3Adam/newDenseLayerOutput/bias/m/Read/ReadVariableOpReadVariableOpAdam/newDenseLayerOutput/bias/m*
_output_shapes
:*
dtype0
?
Adam/newDenseLayer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@*-
shared_nameAdam/newDenseLayer1/kernel/v
?
0Adam/newDenseLayer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer1/kernel/v*
_output_shapes

:*@*
dtype0
?
Adam/newDenseLayer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/newDenseLayer1/bias/v
?
.Adam/newDenseLayer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/newDenseLayer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*-
shared_nameAdam/newDenseLayer2/kernel/v
?
0Adam/newDenseLayer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer2/kernel/v*
_output_shapes
:	@?*
dtype0
?
Adam/newDenseLayer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer2/bias/v
?
.Adam/newDenseLayer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer3/kernel/v
?
0Adam/newDenseLayer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer3/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer3/bias/v
?
.Adam/newDenseLayer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer4/kernel/v
?
0Adam/newDenseLayer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer4/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer4/bias/v
?
.Adam/newDenseLayer4/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer5/kernel/v
?
0Adam/newDenseLayer5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer5/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer5/bias/v
?
.Adam/newDenseLayer5/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer5/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_nameAdam/newDenseLayer6/kernel/v
?
0Adam/newDenseLayer6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer6/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/newDenseLayer6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/newDenseLayer6/bias/v
?
.Adam/newDenseLayer6/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer6/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/newDenseLayer7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_nameAdam/newDenseLayer7/kernel/v
?
0Adam/newDenseLayer7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer7/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/newDenseLayer7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameAdam/newDenseLayer7/bias/v
?
.Adam/newDenseLayer7/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayer7/bias/v*
_output_shapes
:@*
dtype0
?
!Adam/newDenseLayerOutput/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/newDenseLayerOutput/kernel/v
?
5Adam/newDenseLayerOutput/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/newDenseLayerOutput/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/newDenseLayerOutput/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/newDenseLayerOutput/bias/v
?
3Adam/newDenseLayerOutput/bias/v/Read/ReadVariableOpReadVariableOpAdam/newDenseLayerOutput/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?V
value?VB?V B?V
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?
v
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
v
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15
 
?
trainable_variables

Jlayers
Kmetrics
Lnon_trainable_variables
	variables
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics
 
 
 
 
 
?
trainable_variables

Olayers
Pmetrics
Qnon_trainable_variables
	variables
regularization_losses
Rlayer_regularization_losses
Slayer_metrics
a_
VARIABLE_VALUEnewDenseLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Tlayers
Umetrics
Vnon_trainable_variables
	variables
regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
a_
VARIABLE_VALUEnewDenseLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables

Ylayers
Zmetrics
[non_trainable_variables
	variables
regularization_losses
\layer_regularization_losses
]layer_metrics
a_
VARIABLE_VALUEnewDenseLayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
#trainable_variables

^layers
_metrics
`non_trainable_variables
$	variables
%regularization_losses
alayer_regularization_losses
blayer_metrics
a_
VARIABLE_VALUEnewDenseLayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)trainable_variables

clayers
dmetrics
enon_trainable_variables
*	variables
+regularization_losses
flayer_regularization_losses
glayer_metrics
a_
VARIABLE_VALUEnewDenseLayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
/trainable_variables

hlayers
imetrics
jnon_trainable_variables
0	variables
1regularization_losses
klayer_regularization_losses
llayer_metrics
a_
VARIABLE_VALUEnewDenseLayer6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
5trainable_variables

mlayers
nmetrics
onon_trainable_variables
6	variables
7regularization_losses
player_regularization_losses
qlayer_metrics
a_
VARIABLE_VALUEnewDenseLayer7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEnewDenseLayer7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
;trainable_variables

rlayers
smetrics
tnon_trainable_variables
<	variables
=regularization_losses
ulayer_regularization_losses
vlayer_metrics
fd
VARIABLE_VALUEnewDenseLayerOutput/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnewDenseLayerOutput/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
Atrainable_variables

wlayers
xmetrics
ynon_trainable_variables
B	variables
Cregularization_losses
zlayer_regularization_losses
{layer_metrics
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
?
0
1
2
3
4
5
6
7
	8

|0
}1
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
6
	~total
	count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

~0
1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdam/newDenseLayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/newDenseLayerOutput/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayerOutput/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayer7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/newDenseLayer7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/newDenseLayerOutput/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/newDenseLayerOutput/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_flatten_2_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_2_inputnewDenseLayer1/kernelnewDenseLayer1/biasnewDenseLayer2/kernelnewDenseLayer2/biasnewDenseLayer3/kernelnewDenseLayer3/biasnewDenseLayer4/kernelnewDenseLayer4/biasnewDenseLayer5/kernelnewDenseLayer5/biasnewDenseLayer6/kernelnewDenseLayer6/biasnewDenseLayer7/kernelnewDenseLayer7/biasnewDenseLayerOutput/kernelnewDenseLayerOutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_27618
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)newDenseLayer1/kernel/Read/ReadVariableOp'newDenseLayer1/bias/Read/ReadVariableOp)newDenseLayer2/kernel/Read/ReadVariableOp'newDenseLayer2/bias/Read/ReadVariableOp)newDenseLayer3/kernel/Read/ReadVariableOp'newDenseLayer3/bias/Read/ReadVariableOp)newDenseLayer4/kernel/Read/ReadVariableOp'newDenseLayer4/bias/Read/ReadVariableOp)newDenseLayer5/kernel/Read/ReadVariableOp'newDenseLayer5/bias/Read/ReadVariableOp)newDenseLayer6/kernel/Read/ReadVariableOp'newDenseLayer6/bias/Read/ReadVariableOp)newDenseLayer7/kernel/Read/ReadVariableOp'newDenseLayer7/bias/Read/ReadVariableOp.newDenseLayerOutput/kernel/Read/ReadVariableOp,newDenseLayerOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/newDenseLayer1/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer1/bias/m/Read/ReadVariableOp0Adam/newDenseLayer2/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer2/bias/m/Read/ReadVariableOp0Adam/newDenseLayer3/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer3/bias/m/Read/ReadVariableOp0Adam/newDenseLayer4/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer4/bias/m/Read/ReadVariableOp0Adam/newDenseLayer5/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer5/bias/m/Read/ReadVariableOp0Adam/newDenseLayer6/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer6/bias/m/Read/ReadVariableOp0Adam/newDenseLayer7/kernel/m/Read/ReadVariableOp.Adam/newDenseLayer7/bias/m/Read/ReadVariableOp5Adam/newDenseLayerOutput/kernel/m/Read/ReadVariableOp3Adam/newDenseLayerOutput/bias/m/Read/ReadVariableOp0Adam/newDenseLayer1/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer1/bias/v/Read/ReadVariableOp0Adam/newDenseLayer2/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer2/bias/v/Read/ReadVariableOp0Adam/newDenseLayer3/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer3/bias/v/Read/ReadVariableOp0Adam/newDenseLayer4/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer4/bias/v/Read/ReadVariableOp0Adam/newDenseLayer5/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer5/bias/v/Read/ReadVariableOp0Adam/newDenseLayer6/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer6/bias/v/Read/ReadVariableOp0Adam/newDenseLayer7/kernel/v/Read/ReadVariableOp.Adam/newDenseLayer7/bias/v/Read/ReadVariableOp5Adam/newDenseLayerOutput/kernel/v/Read/ReadVariableOp3Adam/newDenseLayerOutput/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_28181
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenewDenseLayer1/kernelnewDenseLayer1/biasnewDenseLayer2/kernelnewDenseLayer2/biasnewDenseLayer3/kernelnewDenseLayer3/biasnewDenseLayer4/kernelnewDenseLayer4/biasnewDenseLayer5/kernelnewDenseLayer5/biasnewDenseLayer6/kernelnewDenseLayer6/biasnewDenseLayer7/kernelnewDenseLayer7/biasnewDenseLayerOutput/kernelnewDenseLayerOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/newDenseLayer1/kernel/mAdam/newDenseLayer1/bias/mAdam/newDenseLayer2/kernel/mAdam/newDenseLayer2/bias/mAdam/newDenseLayer3/kernel/mAdam/newDenseLayer3/bias/mAdam/newDenseLayer4/kernel/mAdam/newDenseLayer4/bias/mAdam/newDenseLayer5/kernel/mAdam/newDenseLayer5/bias/mAdam/newDenseLayer6/kernel/mAdam/newDenseLayer6/bias/mAdam/newDenseLayer7/kernel/mAdam/newDenseLayer7/bias/m!Adam/newDenseLayerOutput/kernel/mAdam/newDenseLayerOutput/bias/mAdam/newDenseLayer1/kernel/vAdam/newDenseLayer1/bias/vAdam/newDenseLayer2/kernel/vAdam/newDenseLayer2/bias/vAdam/newDenseLayer3/kernel/vAdam/newDenseLayer3/bias/vAdam/newDenseLayer4/kernel/vAdam/newDenseLayer4/bias/vAdam/newDenseLayer5/kernel/vAdam/newDenseLayer5/bias/vAdam/newDenseLayer6/kernel/vAdam/newDenseLayer6/bias/vAdam/newDenseLayer7/kernel/vAdam/newDenseLayer7/bias/v!Adam/newDenseLayerOutput/kernel/vAdam/newDenseLayerOutput/bias/v*E
Tin>
<2:*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_28362è	
?
?
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_27097

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer5_layer_call_fn_27916

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_271482
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer3_layer_call_fn_27876

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_271142
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_27528
flatten_2_input&
newdenselayer1_27487:*@"
newdenselayer1_27489:@'
newdenselayer2_27492:	@?#
newdenselayer2_27494:	?(
newdenselayer3_27497:
??#
newdenselayer3_27499:	?(
newdenselayer4_27502:
??#
newdenselayer4_27504:	?(
newdenselayer5_27507:
??#
newdenselayer5_27509:	?(
newdenselayer6_27512:
??#
newdenselayer6_27514:	?'
newdenselayer7_27517:	?@"
newdenselayer7_27519:@+
newdenselayeroutput_27522:@'
newdenselayeroutput_27524:
identity??&newDenseLayer1/StatefulPartitionedCall?&newDenseLayer2/StatefulPartitionedCall?&newDenseLayer3/StatefulPartitionedCall?&newDenseLayer4/StatefulPartitionedCall?&newDenseLayer5/StatefulPartitionedCall?&newDenseLayer6/StatefulPartitionedCall?&newDenseLayer7/StatefulPartitionedCall?+newDenseLayerOutput/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCallflatten_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_270672
flatten_2/PartitionedCall?
&newDenseLayer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0newdenselayer1_27487newdenselayer1_27489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_270802(
&newDenseLayer1/StatefulPartitionedCall?
&newDenseLayer2/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer1/StatefulPartitionedCall:output:0newdenselayer2_27492newdenselayer2_27494*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_270972(
&newDenseLayer2/StatefulPartitionedCall?
&newDenseLayer3/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer2/StatefulPartitionedCall:output:0newdenselayer3_27497newdenselayer3_27499*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_271142(
&newDenseLayer3/StatefulPartitionedCall?
&newDenseLayer4/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer3/StatefulPartitionedCall:output:0newdenselayer4_27502newdenselayer4_27504*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_271312(
&newDenseLayer4/StatefulPartitionedCall?
&newDenseLayer5/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer4/StatefulPartitionedCall:output:0newdenselayer5_27507newdenselayer5_27509*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_271482(
&newDenseLayer5/StatefulPartitionedCall?
&newDenseLayer6/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer5/StatefulPartitionedCall:output:0newdenselayer6_27512newdenselayer6_27514*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_271652(
&newDenseLayer6/StatefulPartitionedCall?
&newDenseLayer7/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer6/StatefulPartitionedCall:output:0newdenselayer7_27517newdenselayer7_27519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_271822(
&newDenseLayer7/StatefulPartitionedCall?
+newDenseLayerOutput/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer7/StatefulPartitionedCall:output:0newdenselayeroutput_27522newdenselayeroutput_27524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_271992-
+newDenseLayerOutput/StatefulPartitionedCall?
IdentityIdentity4newDenseLayerOutput/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^newDenseLayer1/StatefulPartitionedCall'^newDenseLayer2/StatefulPartitionedCall'^newDenseLayer3/StatefulPartitionedCall'^newDenseLayer4/StatefulPartitionedCall'^newDenseLayer5/StatefulPartitionedCall'^newDenseLayer6/StatefulPartitionedCall'^newDenseLayer7/StatefulPartitionedCall,^newDenseLayerOutput/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2P
&newDenseLayer1/StatefulPartitionedCall&newDenseLayer1/StatefulPartitionedCall2P
&newDenseLayer2/StatefulPartitionedCall&newDenseLayer2/StatefulPartitionedCall2P
&newDenseLayer3/StatefulPartitionedCall&newDenseLayer3/StatefulPartitionedCall2P
&newDenseLayer4/StatefulPartitionedCall&newDenseLayer4/StatefulPartitionedCall2P
&newDenseLayer5/StatefulPartitionedCall&newDenseLayer5/StatefulPartitionedCall2P
&newDenseLayer6/StatefulPartitionedCall&newDenseLayer6/StatefulPartitionedCall2P
&newDenseLayer7/StatefulPartitionedCall&newDenseLayer7/StatefulPartitionedCall2Z
+newDenseLayerOutput/StatefulPartitionedCall+newDenseLayerOutput/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?
?
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_27165

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?n
?
 __inference__wrapped_model_27054
flatten_2_inputJ
8sequential_newdenselayer1_matmul_readvariableop_resource:*@G
9sequential_newdenselayer1_biasadd_readvariableop_resource:@K
8sequential_newdenselayer2_matmul_readvariableop_resource:	@?H
9sequential_newdenselayer2_biasadd_readvariableop_resource:	?L
8sequential_newdenselayer3_matmul_readvariableop_resource:
??H
9sequential_newdenselayer3_biasadd_readvariableop_resource:	?L
8sequential_newdenselayer4_matmul_readvariableop_resource:
??H
9sequential_newdenselayer4_biasadd_readvariableop_resource:	?L
8sequential_newdenselayer5_matmul_readvariableop_resource:
??H
9sequential_newdenselayer5_biasadd_readvariableop_resource:	?L
8sequential_newdenselayer6_matmul_readvariableop_resource:
??H
9sequential_newdenselayer6_biasadd_readvariableop_resource:	?K
8sequential_newdenselayer7_matmul_readvariableop_resource:	?@G
9sequential_newdenselayer7_biasadd_readvariableop_resource:@O
=sequential_newdenselayeroutput_matmul_readvariableop_resource:@L
>sequential_newdenselayeroutput_biasadd_readvariableop_resource:
identity??0sequential/newDenseLayer1/BiasAdd/ReadVariableOp?/sequential/newDenseLayer1/MatMul/ReadVariableOp?0sequential/newDenseLayer2/BiasAdd/ReadVariableOp?/sequential/newDenseLayer2/MatMul/ReadVariableOp?0sequential/newDenseLayer3/BiasAdd/ReadVariableOp?/sequential/newDenseLayer3/MatMul/ReadVariableOp?0sequential/newDenseLayer4/BiasAdd/ReadVariableOp?/sequential/newDenseLayer4/MatMul/ReadVariableOp?0sequential/newDenseLayer5/BiasAdd/ReadVariableOp?/sequential/newDenseLayer5/MatMul/ReadVariableOp?0sequential/newDenseLayer6/BiasAdd/ReadVariableOp?/sequential/newDenseLayer6/MatMul/ReadVariableOp?0sequential/newDenseLayer7/BiasAdd/ReadVariableOp?/sequential/newDenseLayer7/MatMul/ReadVariableOp?5sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp?4sequential/newDenseLayerOutput/MatMul/ReadVariableOp?
sequential/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????*   2
sequential/flatten_2/Const?
sequential/flatten_2/ReshapeReshapeflatten_2_input#sequential/flatten_2/Const:output:0*
T0*'
_output_shapes
:?????????*2
sequential/flatten_2/Reshape?
/sequential/newDenseLayer1/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer1_matmul_readvariableop_resource*
_output_shapes

:*@*
dtype021
/sequential/newDenseLayer1/MatMul/ReadVariableOp?
 sequential/newDenseLayer1/MatMulMatMul%sequential/flatten_2/Reshape:output:07sequential/newDenseLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 sequential/newDenseLayer1/MatMul?
0sequential/newDenseLayer1/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential/newDenseLayer1/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer1/BiasAddBiasAdd*sequential/newDenseLayer1/MatMul:product:08sequential/newDenseLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!sequential/newDenseLayer1/BiasAdd?
sequential/newDenseLayer1/ReluRelu*sequential/newDenseLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2 
sequential/newDenseLayer1/Relu?
/sequential/newDenseLayer2/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype021
/sequential/newDenseLayer2/MatMul/ReadVariableOp?
 sequential/newDenseLayer2/MatMulMatMul,sequential/newDenseLayer1/Relu:activations:07sequential/newDenseLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 sequential/newDenseLayer2/MatMul?
0sequential/newDenseLayer2/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0sequential/newDenseLayer2/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer2/BiasAddBiasAdd*sequential/newDenseLayer2/MatMul:product:08sequential/newDenseLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!sequential/newDenseLayer2/BiasAdd?
sequential/newDenseLayer2/ReluRelu*sequential/newDenseLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential/newDenseLayer2/Relu?
/sequential/newDenseLayer3/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential/newDenseLayer3/MatMul/ReadVariableOp?
 sequential/newDenseLayer3/MatMulMatMul,sequential/newDenseLayer2/Relu:activations:07sequential/newDenseLayer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 sequential/newDenseLayer3/MatMul?
0sequential/newDenseLayer3/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0sequential/newDenseLayer3/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer3/BiasAddBiasAdd*sequential/newDenseLayer3/MatMul:product:08sequential/newDenseLayer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!sequential/newDenseLayer3/BiasAdd?
sequential/newDenseLayer3/ReluRelu*sequential/newDenseLayer3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential/newDenseLayer3/Relu?
/sequential/newDenseLayer4/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential/newDenseLayer4/MatMul/ReadVariableOp?
 sequential/newDenseLayer4/MatMulMatMul,sequential/newDenseLayer3/Relu:activations:07sequential/newDenseLayer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 sequential/newDenseLayer4/MatMul?
0sequential/newDenseLayer4/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0sequential/newDenseLayer4/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer4/BiasAddBiasAdd*sequential/newDenseLayer4/MatMul:product:08sequential/newDenseLayer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!sequential/newDenseLayer4/BiasAdd?
sequential/newDenseLayer4/ReluRelu*sequential/newDenseLayer4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential/newDenseLayer4/Relu?
/sequential/newDenseLayer5/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential/newDenseLayer5/MatMul/ReadVariableOp?
 sequential/newDenseLayer5/MatMulMatMul,sequential/newDenseLayer4/Relu:activations:07sequential/newDenseLayer5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 sequential/newDenseLayer5/MatMul?
0sequential/newDenseLayer5/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0sequential/newDenseLayer5/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer5/BiasAddBiasAdd*sequential/newDenseLayer5/MatMul:product:08sequential/newDenseLayer5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!sequential/newDenseLayer5/BiasAdd?
sequential/newDenseLayer5/ReluRelu*sequential/newDenseLayer5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential/newDenseLayer5/Relu?
/sequential/newDenseLayer6/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential/newDenseLayer6/MatMul/ReadVariableOp?
 sequential/newDenseLayer6/MatMulMatMul,sequential/newDenseLayer5/Relu:activations:07sequential/newDenseLayer6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 sequential/newDenseLayer6/MatMul?
0sequential/newDenseLayer6/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0sequential/newDenseLayer6/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer6/BiasAddBiasAdd*sequential/newDenseLayer6/MatMul:product:08sequential/newDenseLayer6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!sequential/newDenseLayer6/BiasAdd?
sequential/newDenseLayer6/ReluRelu*sequential/newDenseLayer6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential/newDenseLayer6/Relu?
/sequential/newDenseLayer7/MatMul/ReadVariableOpReadVariableOp8sequential_newdenselayer7_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype021
/sequential/newDenseLayer7/MatMul/ReadVariableOp?
 sequential/newDenseLayer7/MatMulMatMul,sequential/newDenseLayer6/Relu:activations:07sequential/newDenseLayer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 sequential/newDenseLayer7/MatMul?
0sequential/newDenseLayer7/BiasAdd/ReadVariableOpReadVariableOp9sequential_newdenselayer7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential/newDenseLayer7/BiasAdd/ReadVariableOp?
!sequential/newDenseLayer7/BiasAddBiasAdd*sequential/newDenseLayer7/MatMul:product:08sequential/newDenseLayer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!sequential/newDenseLayer7/BiasAdd?
sequential/newDenseLayer7/ReluRelu*sequential/newDenseLayer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2 
sequential/newDenseLayer7/Relu?
4sequential/newDenseLayerOutput/MatMul/ReadVariableOpReadVariableOp=sequential_newdenselayeroutput_matmul_readvariableop_resource*
_output_shapes

:@*
dtype026
4sequential/newDenseLayerOutput/MatMul/ReadVariableOp?
%sequential/newDenseLayerOutput/MatMulMatMul,sequential/newDenseLayer7/Relu:activations:0<sequential/newDenseLayerOutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%sequential/newDenseLayerOutput/MatMul?
5sequential/newDenseLayerOutput/BiasAdd/ReadVariableOpReadVariableOp>sequential_newdenselayeroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp?
&sequential/newDenseLayerOutput/BiasAddBiasAdd/sequential/newDenseLayerOutput/MatMul:product:0=sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&sequential/newDenseLayerOutput/BiasAdd?
&sequential/newDenseLayerOutput/SoftmaxSoftmax/sequential/newDenseLayerOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2(
&sequential/newDenseLayerOutput/Softmax?
IdentityIdentity0sequential/newDenseLayerOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp1^sequential/newDenseLayer1/BiasAdd/ReadVariableOp0^sequential/newDenseLayer1/MatMul/ReadVariableOp1^sequential/newDenseLayer2/BiasAdd/ReadVariableOp0^sequential/newDenseLayer2/MatMul/ReadVariableOp1^sequential/newDenseLayer3/BiasAdd/ReadVariableOp0^sequential/newDenseLayer3/MatMul/ReadVariableOp1^sequential/newDenseLayer4/BiasAdd/ReadVariableOp0^sequential/newDenseLayer4/MatMul/ReadVariableOp1^sequential/newDenseLayer5/BiasAdd/ReadVariableOp0^sequential/newDenseLayer5/MatMul/ReadVariableOp1^sequential/newDenseLayer6/BiasAdd/ReadVariableOp0^sequential/newDenseLayer6/MatMul/ReadVariableOp1^sequential/newDenseLayer7/BiasAdd/ReadVariableOp0^sequential/newDenseLayer7/MatMul/ReadVariableOp6^sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp5^sequential/newDenseLayerOutput/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2d
0sequential/newDenseLayer1/BiasAdd/ReadVariableOp0sequential/newDenseLayer1/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer1/MatMul/ReadVariableOp/sequential/newDenseLayer1/MatMul/ReadVariableOp2d
0sequential/newDenseLayer2/BiasAdd/ReadVariableOp0sequential/newDenseLayer2/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer2/MatMul/ReadVariableOp/sequential/newDenseLayer2/MatMul/ReadVariableOp2d
0sequential/newDenseLayer3/BiasAdd/ReadVariableOp0sequential/newDenseLayer3/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer3/MatMul/ReadVariableOp/sequential/newDenseLayer3/MatMul/ReadVariableOp2d
0sequential/newDenseLayer4/BiasAdd/ReadVariableOp0sequential/newDenseLayer4/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer4/MatMul/ReadVariableOp/sequential/newDenseLayer4/MatMul/ReadVariableOp2d
0sequential/newDenseLayer5/BiasAdd/ReadVariableOp0sequential/newDenseLayer5/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer5/MatMul/ReadVariableOp/sequential/newDenseLayer5/MatMul/ReadVariableOp2d
0sequential/newDenseLayer6/BiasAdd/ReadVariableOp0sequential/newDenseLayer6/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer6/MatMul/ReadVariableOp/sequential/newDenseLayer6/MatMul/ReadVariableOp2d
0sequential/newDenseLayer7/BiasAdd/ReadVariableOp0sequential/newDenseLayer7/BiasAdd/ReadVariableOp2b
/sequential/newDenseLayer7/MatMul/ReadVariableOp/sequential/newDenseLayer7/MatMul/ReadVariableOp2n
5sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp5sequential/newDenseLayerOutput/BiasAdd/ReadVariableOp2l
4sequential/newDenseLayerOutput/MatMul/ReadVariableOp4sequential/newDenseLayerOutput/MatMul/ReadVariableOp:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?
?
*__inference_sequential_layer_call_fn_27241
flatten_2_input
unknown:*@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_272062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?
?
*__inference_sequential_layer_call_fn_27655

inputs
unknown:*@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_272062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_27967

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_27067

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????*   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????*2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_27411

inputs&
newdenselayer1_27370:*@"
newdenselayer1_27372:@'
newdenselayer2_27375:	@?#
newdenselayer2_27377:	?(
newdenselayer3_27380:
??#
newdenselayer3_27382:	?(
newdenselayer4_27385:
??#
newdenselayer4_27387:	?(
newdenselayer5_27390:
??#
newdenselayer5_27392:	?(
newdenselayer6_27395:
??#
newdenselayer6_27397:	?'
newdenselayer7_27400:	?@"
newdenselayer7_27402:@+
newdenselayeroutput_27405:@'
newdenselayeroutput_27407:
identity??&newDenseLayer1/StatefulPartitionedCall?&newDenseLayer2/StatefulPartitionedCall?&newDenseLayer3/StatefulPartitionedCall?&newDenseLayer4/StatefulPartitionedCall?&newDenseLayer5/StatefulPartitionedCall?&newDenseLayer6/StatefulPartitionedCall?&newDenseLayer7/StatefulPartitionedCall?+newDenseLayerOutput/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_270672
flatten_2/PartitionedCall?
&newDenseLayer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0newdenselayer1_27370newdenselayer1_27372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_270802(
&newDenseLayer1/StatefulPartitionedCall?
&newDenseLayer2/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer1/StatefulPartitionedCall:output:0newdenselayer2_27375newdenselayer2_27377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_270972(
&newDenseLayer2/StatefulPartitionedCall?
&newDenseLayer3/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer2/StatefulPartitionedCall:output:0newdenselayer3_27380newdenselayer3_27382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_271142(
&newDenseLayer3/StatefulPartitionedCall?
&newDenseLayer4/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer3/StatefulPartitionedCall:output:0newdenselayer4_27385newdenselayer4_27387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_271312(
&newDenseLayer4/StatefulPartitionedCall?
&newDenseLayer5/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer4/StatefulPartitionedCall:output:0newdenselayer5_27390newdenselayer5_27392*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_271482(
&newDenseLayer5/StatefulPartitionedCall?
&newDenseLayer6/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer5/StatefulPartitionedCall:output:0newdenselayer6_27395newdenselayer6_27397*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_271652(
&newDenseLayer6/StatefulPartitionedCall?
&newDenseLayer7/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer6/StatefulPartitionedCall:output:0newdenselayer7_27400newdenselayer7_27402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_271822(
&newDenseLayer7/StatefulPartitionedCall?
+newDenseLayerOutput/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer7/StatefulPartitionedCall:output:0newdenselayeroutput_27405newdenselayeroutput_27407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_271992-
+newDenseLayerOutput/StatefulPartitionedCall?
IdentityIdentity4newDenseLayerOutput/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^newDenseLayer1/StatefulPartitionedCall'^newDenseLayer2/StatefulPartitionedCall'^newDenseLayer3/StatefulPartitionedCall'^newDenseLayer4/StatefulPartitionedCall'^newDenseLayer5/StatefulPartitionedCall'^newDenseLayer6/StatefulPartitionedCall'^newDenseLayer7/StatefulPartitionedCall,^newDenseLayerOutput/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2P
&newDenseLayer1/StatefulPartitionedCall&newDenseLayer1/StatefulPartitionedCall2P
&newDenseLayer2/StatefulPartitionedCall&newDenseLayer2/StatefulPartitionedCall2P
&newDenseLayer3/StatefulPartitionedCall&newDenseLayer3/StatefulPartitionedCall2P
&newDenseLayer4/StatefulPartitionedCall&newDenseLayer4/StatefulPartitionedCall2P
&newDenseLayer5/StatefulPartitionedCall&newDenseLayer5/StatefulPartitionedCall2P
&newDenseLayer6/StatefulPartitionedCall&newDenseLayer6/StatefulPartitionedCall2P
&newDenseLayer7/StatefulPartitionedCall&newDenseLayer7/StatefulPartitionedCall2Z
+newDenseLayerOutput/StatefulPartitionedCall+newDenseLayerOutput/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_27483
flatten_2_input
unknown:*@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_274112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?[
?
E__inference_sequential_layer_call_and_return_conditional_losses_27754

inputs?
-newdenselayer1_matmul_readvariableop_resource:*@<
.newdenselayer1_biasadd_readvariableop_resource:@@
-newdenselayer2_matmul_readvariableop_resource:	@?=
.newdenselayer2_biasadd_readvariableop_resource:	?A
-newdenselayer3_matmul_readvariableop_resource:
??=
.newdenselayer3_biasadd_readvariableop_resource:	?A
-newdenselayer4_matmul_readvariableop_resource:
??=
.newdenselayer4_biasadd_readvariableop_resource:	?A
-newdenselayer5_matmul_readvariableop_resource:
??=
.newdenselayer5_biasadd_readvariableop_resource:	?A
-newdenselayer6_matmul_readvariableop_resource:
??=
.newdenselayer6_biasadd_readvariableop_resource:	?@
-newdenselayer7_matmul_readvariableop_resource:	?@<
.newdenselayer7_biasadd_readvariableop_resource:@D
2newdenselayeroutput_matmul_readvariableop_resource:@A
3newdenselayeroutput_biasadd_readvariableop_resource:
identity??%newDenseLayer1/BiasAdd/ReadVariableOp?$newDenseLayer1/MatMul/ReadVariableOp?%newDenseLayer2/BiasAdd/ReadVariableOp?$newDenseLayer2/MatMul/ReadVariableOp?%newDenseLayer3/BiasAdd/ReadVariableOp?$newDenseLayer3/MatMul/ReadVariableOp?%newDenseLayer4/BiasAdd/ReadVariableOp?$newDenseLayer4/MatMul/ReadVariableOp?%newDenseLayer5/BiasAdd/ReadVariableOp?$newDenseLayer5/MatMul/ReadVariableOp?%newDenseLayer6/BiasAdd/ReadVariableOp?$newDenseLayer6/MatMul/ReadVariableOp?%newDenseLayer7/BiasAdd/ReadVariableOp?$newDenseLayer7/MatMul/ReadVariableOp?*newDenseLayerOutput/BiasAdd/ReadVariableOp?)newDenseLayerOutput/MatMul/ReadVariableOps
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????*   2
flatten_2/Const?
flatten_2/ReshapeReshapeinputsflatten_2/Const:output:0*
T0*'
_output_shapes
:?????????*2
flatten_2/Reshape?
$newDenseLayer1/MatMul/ReadVariableOpReadVariableOp-newdenselayer1_matmul_readvariableop_resource*
_output_shapes

:*@*
dtype02&
$newDenseLayer1/MatMul/ReadVariableOp?
newDenseLayer1/MatMulMatMulflatten_2/Reshape:output:0,newDenseLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/MatMul?
%newDenseLayer1/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%newDenseLayer1/BiasAdd/ReadVariableOp?
newDenseLayer1/BiasAddBiasAddnewDenseLayer1/MatMul:product:0-newDenseLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/BiasAdd?
newDenseLayer1/ReluRelunewDenseLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/Relu?
$newDenseLayer2/MatMul/ReadVariableOpReadVariableOp-newdenselayer2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$newDenseLayer2/MatMul/ReadVariableOp?
newDenseLayer2/MatMulMatMul!newDenseLayer1/Relu:activations:0,newDenseLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/MatMul?
%newDenseLayer2/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer2/BiasAdd/ReadVariableOp?
newDenseLayer2/BiasAddBiasAddnewDenseLayer2/MatMul:product:0-newDenseLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/BiasAdd?
newDenseLayer2/ReluRelunewDenseLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/Relu?
$newDenseLayer3/MatMul/ReadVariableOpReadVariableOp-newdenselayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer3/MatMul/ReadVariableOp?
newDenseLayer3/MatMulMatMul!newDenseLayer2/Relu:activations:0,newDenseLayer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/MatMul?
%newDenseLayer3/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer3/BiasAdd/ReadVariableOp?
newDenseLayer3/BiasAddBiasAddnewDenseLayer3/MatMul:product:0-newDenseLayer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/BiasAdd?
newDenseLayer3/ReluRelunewDenseLayer3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/Relu?
$newDenseLayer4/MatMul/ReadVariableOpReadVariableOp-newdenselayer4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer4/MatMul/ReadVariableOp?
newDenseLayer4/MatMulMatMul!newDenseLayer3/Relu:activations:0,newDenseLayer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/MatMul?
%newDenseLayer4/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer4/BiasAdd/ReadVariableOp?
newDenseLayer4/BiasAddBiasAddnewDenseLayer4/MatMul:product:0-newDenseLayer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/BiasAdd?
newDenseLayer4/ReluRelunewDenseLayer4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/Relu?
$newDenseLayer5/MatMul/ReadVariableOpReadVariableOp-newdenselayer5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer5/MatMul/ReadVariableOp?
newDenseLayer5/MatMulMatMul!newDenseLayer4/Relu:activations:0,newDenseLayer5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/MatMul?
%newDenseLayer5/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer5/BiasAdd/ReadVariableOp?
newDenseLayer5/BiasAddBiasAddnewDenseLayer5/MatMul:product:0-newDenseLayer5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/BiasAdd?
newDenseLayer5/ReluRelunewDenseLayer5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/Relu?
$newDenseLayer6/MatMul/ReadVariableOpReadVariableOp-newdenselayer6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer6/MatMul/ReadVariableOp?
newDenseLayer6/MatMulMatMul!newDenseLayer5/Relu:activations:0,newDenseLayer6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/MatMul?
%newDenseLayer6/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer6/BiasAdd/ReadVariableOp?
newDenseLayer6/BiasAddBiasAddnewDenseLayer6/MatMul:product:0-newDenseLayer6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/BiasAdd?
newDenseLayer6/ReluRelunewDenseLayer6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/Relu?
$newDenseLayer7/MatMul/ReadVariableOpReadVariableOp-newdenselayer7_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02&
$newDenseLayer7/MatMul/ReadVariableOp?
newDenseLayer7/MatMulMatMul!newDenseLayer6/Relu:activations:0,newDenseLayer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/MatMul?
%newDenseLayer7/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%newDenseLayer7/BiasAdd/ReadVariableOp?
newDenseLayer7/BiasAddBiasAddnewDenseLayer7/MatMul:product:0-newDenseLayer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/BiasAdd?
newDenseLayer7/ReluRelunewDenseLayer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/Relu?
)newDenseLayerOutput/MatMul/ReadVariableOpReadVariableOp2newdenselayeroutput_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)newDenseLayerOutput/MatMul/ReadVariableOp?
newDenseLayerOutput/MatMulMatMul!newDenseLayer7/Relu:activations:01newDenseLayerOutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/MatMul?
*newDenseLayerOutput/BiasAdd/ReadVariableOpReadVariableOp3newdenselayeroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*newDenseLayerOutput/BiasAdd/ReadVariableOp?
newDenseLayerOutput/BiasAddBiasAdd$newDenseLayerOutput/MatMul:product:02newDenseLayerOutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/BiasAdd?
newDenseLayerOutput/SoftmaxSoftmax$newDenseLayerOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/Softmax?
IdentityIdentity%newDenseLayerOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^newDenseLayer1/BiasAdd/ReadVariableOp%^newDenseLayer1/MatMul/ReadVariableOp&^newDenseLayer2/BiasAdd/ReadVariableOp%^newDenseLayer2/MatMul/ReadVariableOp&^newDenseLayer3/BiasAdd/ReadVariableOp%^newDenseLayer3/MatMul/ReadVariableOp&^newDenseLayer4/BiasAdd/ReadVariableOp%^newDenseLayer4/MatMul/ReadVariableOp&^newDenseLayer5/BiasAdd/ReadVariableOp%^newDenseLayer5/MatMul/ReadVariableOp&^newDenseLayer6/BiasAdd/ReadVariableOp%^newDenseLayer6/MatMul/ReadVariableOp&^newDenseLayer7/BiasAdd/ReadVariableOp%^newDenseLayer7/MatMul/ReadVariableOp+^newDenseLayerOutput/BiasAdd/ReadVariableOp*^newDenseLayerOutput/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2N
%newDenseLayer1/BiasAdd/ReadVariableOp%newDenseLayer1/BiasAdd/ReadVariableOp2L
$newDenseLayer1/MatMul/ReadVariableOp$newDenseLayer1/MatMul/ReadVariableOp2N
%newDenseLayer2/BiasAdd/ReadVariableOp%newDenseLayer2/BiasAdd/ReadVariableOp2L
$newDenseLayer2/MatMul/ReadVariableOp$newDenseLayer2/MatMul/ReadVariableOp2N
%newDenseLayer3/BiasAdd/ReadVariableOp%newDenseLayer3/BiasAdd/ReadVariableOp2L
$newDenseLayer3/MatMul/ReadVariableOp$newDenseLayer3/MatMul/ReadVariableOp2N
%newDenseLayer4/BiasAdd/ReadVariableOp%newDenseLayer4/BiasAdd/ReadVariableOp2L
$newDenseLayer4/MatMul/ReadVariableOp$newDenseLayer4/MatMul/ReadVariableOp2N
%newDenseLayer5/BiasAdd/ReadVariableOp%newDenseLayer5/BiasAdd/ReadVariableOp2L
$newDenseLayer5/MatMul/ReadVariableOp$newDenseLayer5/MatMul/ReadVariableOp2N
%newDenseLayer6/BiasAdd/ReadVariableOp%newDenseLayer6/BiasAdd/ReadVariableOp2L
$newDenseLayer6/MatMul/ReadVariableOp$newDenseLayer6/MatMul/ReadVariableOp2N
%newDenseLayer7/BiasAdd/ReadVariableOp%newDenseLayer7/BiasAdd/ReadVariableOp2L
$newDenseLayer7/MatMul/ReadVariableOp$newDenseLayer7/MatMul/ReadVariableOp2X
*newDenseLayerOutput/BiasAdd/ReadVariableOp*newDenseLayerOutput/BiasAdd/ReadVariableOp2V
)newDenseLayerOutput/MatMul/ReadVariableOp)newDenseLayerOutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_27827

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????*   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????*2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?[
?
E__inference_sequential_layer_call_and_return_conditional_losses_27816

inputs?
-newdenselayer1_matmul_readvariableop_resource:*@<
.newdenselayer1_biasadd_readvariableop_resource:@@
-newdenselayer2_matmul_readvariableop_resource:	@?=
.newdenselayer2_biasadd_readvariableop_resource:	?A
-newdenselayer3_matmul_readvariableop_resource:
??=
.newdenselayer3_biasadd_readvariableop_resource:	?A
-newdenselayer4_matmul_readvariableop_resource:
??=
.newdenselayer4_biasadd_readvariableop_resource:	?A
-newdenselayer5_matmul_readvariableop_resource:
??=
.newdenselayer5_biasadd_readvariableop_resource:	?A
-newdenselayer6_matmul_readvariableop_resource:
??=
.newdenselayer6_biasadd_readvariableop_resource:	?@
-newdenselayer7_matmul_readvariableop_resource:	?@<
.newdenselayer7_biasadd_readvariableop_resource:@D
2newdenselayeroutput_matmul_readvariableop_resource:@A
3newdenselayeroutput_biasadd_readvariableop_resource:
identity??%newDenseLayer1/BiasAdd/ReadVariableOp?$newDenseLayer1/MatMul/ReadVariableOp?%newDenseLayer2/BiasAdd/ReadVariableOp?$newDenseLayer2/MatMul/ReadVariableOp?%newDenseLayer3/BiasAdd/ReadVariableOp?$newDenseLayer3/MatMul/ReadVariableOp?%newDenseLayer4/BiasAdd/ReadVariableOp?$newDenseLayer4/MatMul/ReadVariableOp?%newDenseLayer5/BiasAdd/ReadVariableOp?$newDenseLayer5/MatMul/ReadVariableOp?%newDenseLayer6/BiasAdd/ReadVariableOp?$newDenseLayer6/MatMul/ReadVariableOp?%newDenseLayer7/BiasAdd/ReadVariableOp?$newDenseLayer7/MatMul/ReadVariableOp?*newDenseLayerOutput/BiasAdd/ReadVariableOp?)newDenseLayerOutput/MatMul/ReadVariableOps
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????*   2
flatten_2/Const?
flatten_2/ReshapeReshapeinputsflatten_2/Const:output:0*
T0*'
_output_shapes
:?????????*2
flatten_2/Reshape?
$newDenseLayer1/MatMul/ReadVariableOpReadVariableOp-newdenselayer1_matmul_readvariableop_resource*
_output_shapes

:*@*
dtype02&
$newDenseLayer1/MatMul/ReadVariableOp?
newDenseLayer1/MatMulMatMulflatten_2/Reshape:output:0,newDenseLayer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/MatMul?
%newDenseLayer1/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%newDenseLayer1/BiasAdd/ReadVariableOp?
newDenseLayer1/BiasAddBiasAddnewDenseLayer1/MatMul:product:0-newDenseLayer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/BiasAdd?
newDenseLayer1/ReluRelunewDenseLayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer1/Relu?
$newDenseLayer2/MatMul/ReadVariableOpReadVariableOp-newdenselayer2_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02&
$newDenseLayer2/MatMul/ReadVariableOp?
newDenseLayer2/MatMulMatMul!newDenseLayer1/Relu:activations:0,newDenseLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/MatMul?
%newDenseLayer2/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer2/BiasAdd/ReadVariableOp?
newDenseLayer2/BiasAddBiasAddnewDenseLayer2/MatMul:product:0-newDenseLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/BiasAdd?
newDenseLayer2/ReluRelunewDenseLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer2/Relu?
$newDenseLayer3/MatMul/ReadVariableOpReadVariableOp-newdenselayer3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer3/MatMul/ReadVariableOp?
newDenseLayer3/MatMulMatMul!newDenseLayer2/Relu:activations:0,newDenseLayer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/MatMul?
%newDenseLayer3/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer3/BiasAdd/ReadVariableOp?
newDenseLayer3/BiasAddBiasAddnewDenseLayer3/MatMul:product:0-newDenseLayer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/BiasAdd?
newDenseLayer3/ReluRelunewDenseLayer3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer3/Relu?
$newDenseLayer4/MatMul/ReadVariableOpReadVariableOp-newdenselayer4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer4/MatMul/ReadVariableOp?
newDenseLayer4/MatMulMatMul!newDenseLayer3/Relu:activations:0,newDenseLayer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/MatMul?
%newDenseLayer4/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer4/BiasAdd/ReadVariableOp?
newDenseLayer4/BiasAddBiasAddnewDenseLayer4/MatMul:product:0-newDenseLayer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/BiasAdd?
newDenseLayer4/ReluRelunewDenseLayer4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer4/Relu?
$newDenseLayer5/MatMul/ReadVariableOpReadVariableOp-newdenselayer5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer5/MatMul/ReadVariableOp?
newDenseLayer5/MatMulMatMul!newDenseLayer4/Relu:activations:0,newDenseLayer5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/MatMul?
%newDenseLayer5/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer5/BiasAdd/ReadVariableOp?
newDenseLayer5/BiasAddBiasAddnewDenseLayer5/MatMul:product:0-newDenseLayer5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/BiasAdd?
newDenseLayer5/ReluRelunewDenseLayer5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer5/Relu?
$newDenseLayer6/MatMul/ReadVariableOpReadVariableOp-newdenselayer6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$newDenseLayer6/MatMul/ReadVariableOp?
newDenseLayer6/MatMulMatMul!newDenseLayer5/Relu:activations:0,newDenseLayer6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/MatMul?
%newDenseLayer6/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%newDenseLayer6/BiasAdd/ReadVariableOp?
newDenseLayer6/BiasAddBiasAddnewDenseLayer6/MatMul:product:0-newDenseLayer6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/BiasAdd?
newDenseLayer6/ReluRelunewDenseLayer6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
newDenseLayer6/Relu?
$newDenseLayer7/MatMul/ReadVariableOpReadVariableOp-newdenselayer7_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02&
$newDenseLayer7/MatMul/ReadVariableOp?
newDenseLayer7/MatMulMatMul!newDenseLayer6/Relu:activations:0,newDenseLayer7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/MatMul?
%newDenseLayer7/BiasAdd/ReadVariableOpReadVariableOp.newdenselayer7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%newDenseLayer7/BiasAdd/ReadVariableOp?
newDenseLayer7/BiasAddBiasAddnewDenseLayer7/MatMul:product:0-newDenseLayer7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/BiasAdd?
newDenseLayer7/ReluRelunewDenseLayer7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
newDenseLayer7/Relu?
)newDenseLayerOutput/MatMul/ReadVariableOpReadVariableOp2newdenselayeroutput_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)newDenseLayerOutput/MatMul/ReadVariableOp?
newDenseLayerOutput/MatMulMatMul!newDenseLayer7/Relu:activations:01newDenseLayerOutput/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/MatMul?
*newDenseLayerOutput/BiasAdd/ReadVariableOpReadVariableOp3newdenselayeroutput_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*newDenseLayerOutput/BiasAdd/ReadVariableOp?
newDenseLayerOutput/BiasAddBiasAdd$newDenseLayerOutput/MatMul:product:02newDenseLayerOutput/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/BiasAdd?
newDenseLayerOutput/SoftmaxSoftmax$newDenseLayerOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
newDenseLayerOutput/Softmax?
IdentityIdentity%newDenseLayerOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^newDenseLayer1/BiasAdd/ReadVariableOp%^newDenseLayer1/MatMul/ReadVariableOp&^newDenseLayer2/BiasAdd/ReadVariableOp%^newDenseLayer2/MatMul/ReadVariableOp&^newDenseLayer3/BiasAdd/ReadVariableOp%^newDenseLayer3/MatMul/ReadVariableOp&^newDenseLayer4/BiasAdd/ReadVariableOp%^newDenseLayer4/MatMul/ReadVariableOp&^newDenseLayer5/BiasAdd/ReadVariableOp%^newDenseLayer5/MatMul/ReadVariableOp&^newDenseLayer6/BiasAdd/ReadVariableOp%^newDenseLayer6/MatMul/ReadVariableOp&^newDenseLayer7/BiasAdd/ReadVariableOp%^newDenseLayer7/MatMul/ReadVariableOp+^newDenseLayerOutput/BiasAdd/ReadVariableOp*^newDenseLayerOutput/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2N
%newDenseLayer1/BiasAdd/ReadVariableOp%newDenseLayer1/BiasAdd/ReadVariableOp2L
$newDenseLayer1/MatMul/ReadVariableOp$newDenseLayer1/MatMul/ReadVariableOp2N
%newDenseLayer2/BiasAdd/ReadVariableOp%newDenseLayer2/BiasAdd/ReadVariableOp2L
$newDenseLayer2/MatMul/ReadVariableOp$newDenseLayer2/MatMul/ReadVariableOp2N
%newDenseLayer3/BiasAdd/ReadVariableOp%newDenseLayer3/BiasAdd/ReadVariableOp2L
$newDenseLayer3/MatMul/ReadVariableOp$newDenseLayer3/MatMul/ReadVariableOp2N
%newDenseLayer4/BiasAdd/ReadVariableOp%newDenseLayer4/BiasAdd/ReadVariableOp2L
$newDenseLayer4/MatMul/ReadVariableOp$newDenseLayer4/MatMul/ReadVariableOp2N
%newDenseLayer5/BiasAdd/ReadVariableOp%newDenseLayer5/BiasAdd/ReadVariableOp2L
$newDenseLayer5/MatMul/ReadVariableOp$newDenseLayer5/MatMul/ReadVariableOp2N
%newDenseLayer6/BiasAdd/ReadVariableOp%newDenseLayer6/BiasAdd/ReadVariableOp2L
$newDenseLayer6/MatMul/ReadVariableOp$newDenseLayer6/MatMul/ReadVariableOp2N
%newDenseLayer7/BiasAdd/ReadVariableOp%newDenseLayer7/BiasAdd/ReadVariableOp2L
$newDenseLayer7/MatMul/ReadVariableOp$newDenseLayer7/MatMul/ReadVariableOp2X
*newDenseLayerOutput/BiasAdd/ReadVariableOp*newDenseLayerOutput/BiasAdd/ReadVariableOp2V
)newDenseLayerOutput/MatMul/ReadVariableOp)newDenseLayerOutput/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_27114

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_27887

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_27821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_270672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????*2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer7_layer_call_fn_27956

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_271822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_27927

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_newDenseLayerOutput_layer_call_fn_27976

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_271992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_27947

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_27573
flatten_2_input&
newdenselayer1_27532:*@"
newdenselayer1_27534:@'
newdenselayer2_27537:	@?#
newdenselayer2_27539:	?(
newdenselayer3_27542:
??#
newdenselayer3_27544:	?(
newdenselayer4_27547:
??#
newdenselayer4_27549:	?(
newdenselayer5_27552:
??#
newdenselayer5_27554:	?(
newdenselayer6_27557:
??#
newdenselayer6_27559:	?'
newdenselayer7_27562:	?@"
newdenselayer7_27564:@+
newdenselayeroutput_27567:@'
newdenselayeroutput_27569:
identity??&newDenseLayer1/StatefulPartitionedCall?&newDenseLayer2/StatefulPartitionedCall?&newDenseLayer3/StatefulPartitionedCall?&newDenseLayer4/StatefulPartitionedCall?&newDenseLayer5/StatefulPartitionedCall?&newDenseLayer6/StatefulPartitionedCall?&newDenseLayer7/StatefulPartitionedCall?+newDenseLayerOutput/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCallflatten_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_270672
flatten_2/PartitionedCall?
&newDenseLayer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0newdenselayer1_27532newdenselayer1_27534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_270802(
&newDenseLayer1/StatefulPartitionedCall?
&newDenseLayer2/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer1/StatefulPartitionedCall:output:0newdenselayer2_27537newdenselayer2_27539*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_270972(
&newDenseLayer2/StatefulPartitionedCall?
&newDenseLayer3/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer2/StatefulPartitionedCall:output:0newdenselayer3_27542newdenselayer3_27544*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_271142(
&newDenseLayer3/StatefulPartitionedCall?
&newDenseLayer4/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer3/StatefulPartitionedCall:output:0newdenselayer4_27547newdenselayer4_27549*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_271312(
&newDenseLayer4/StatefulPartitionedCall?
&newDenseLayer5/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer4/StatefulPartitionedCall:output:0newdenselayer5_27552newdenselayer5_27554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_271482(
&newDenseLayer5/StatefulPartitionedCall?
&newDenseLayer6/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer5/StatefulPartitionedCall:output:0newdenselayer6_27557newdenselayer6_27559*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_271652(
&newDenseLayer6/StatefulPartitionedCall?
&newDenseLayer7/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer6/StatefulPartitionedCall:output:0newdenselayer7_27562newdenselayer7_27564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_271822(
&newDenseLayer7/StatefulPartitionedCall?
+newDenseLayerOutput/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer7/StatefulPartitionedCall:output:0newdenselayeroutput_27567newdenselayeroutput_27569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_271992-
+newDenseLayerOutput/StatefulPartitionedCall?
IdentityIdentity4newDenseLayerOutput/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^newDenseLayer1/StatefulPartitionedCall'^newDenseLayer2/StatefulPartitionedCall'^newDenseLayer3/StatefulPartitionedCall'^newDenseLayer4/StatefulPartitionedCall'^newDenseLayer5/StatefulPartitionedCall'^newDenseLayer6/StatefulPartitionedCall'^newDenseLayer7/StatefulPartitionedCall,^newDenseLayerOutput/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2P
&newDenseLayer1/StatefulPartitionedCall&newDenseLayer1/StatefulPartitionedCall2P
&newDenseLayer2/StatefulPartitionedCall&newDenseLayer2/StatefulPartitionedCall2P
&newDenseLayer3/StatefulPartitionedCall&newDenseLayer3/StatefulPartitionedCall2P
&newDenseLayer4/StatefulPartitionedCall&newDenseLayer4/StatefulPartitionedCall2P
&newDenseLayer5/StatefulPartitionedCall&newDenseLayer5/StatefulPartitionedCall2P
&newDenseLayer6/StatefulPartitionedCall&newDenseLayer6/StatefulPartitionedCall2P
&newDenseLayer7/StatefulPartitionedCall&newDenseLayer7/StatefulPartitionedCall2Z
+newDenseLayerOutput/StatefulPartitionedCall+newDenseLayerOutput/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?w
?
__inference__traced_save_28181
file_prefix4
0savev2_newdenselayer1_kernel_read_readvariableop2
.savev2_newdenselayer1_bias_read_readvariableop4
0savev2_newdenselayer2_kernel_read_readvariableop2
.savev2_newdenselayer2_bias_read_readvariableop4
0savev2_newdenselayer3_kernel_read_readvariableop2
.savev2_newdenselayer3_bias_read_readvariableop4
0savev2_newdenselayer4_kernel_read_readvariableop2
.savev2_newdenselayer4_bias_read_readvariableop4
0savev2_newdenselayer5_kernel_read_readvariableop2
.savev2_newdenselayer5_bias_read_readvariableop4
0savev2_newdenselayer6_kernel_read_readvariableop2
.savev2_newdenselayer6_bias_read_readvariableop4
0savev2_newdenselayer7_kernel_read_readvariableop2
.savev2_newdenselayer7_bias_read_readvariableop9
5savev2_newdenselayeroutput_kernel_read_readvariableop7
3savev2_newdenselayeroutput_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_newdenselayer1_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer1_bias_m_read_readvariableop;
7savev2_adam_newdenselayer2_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer2_bias_m_read_readvariableop;
7savev2_adam_newdenselayer3_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer3_bias_m_read_readvariableop;
7savev2_adam_newdenselayer4_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer4_bias_m_read_readvariableop;
7savev2_adam_newdenselayer5_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer5_bias_m_read_readvariableop;
7savev2_adam_newdenselayer6_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer6_bias_m_read_readvariableop;
7savev2_adam_newdenselayer7_kernel_m_read_readvariableop9
5savev2_adam_newdenselayer7_bias_m_read_readvariableop@
<savev2_adam_newdenselayeroutput_kernel_m_read_readvariableop>
:savev2_adam_newdenselayeroutput_bias_m_read_readvariableop;
7savev2_adam_newdenselayer1_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer1_bias_v_read_readvariableop;
7savev2_adam_newdenselayer2_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer2_bias_v_read_readvariableop;
7savev2_adam_newdenselayer3_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer3_bias_v_read_readvariableop;
7savev2_adam_newdenselayer4_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer4_bias_v_read_readvariableop;
7savev2_adam_newdenselayer5_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer5_bias_v_read_readvariableop;
7savev2_adam_newdenselayer6_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer6_bias_v_read_readvariableop;
7savev2_adam_newdenselayer7_kernel_v_read_readvariableop9
5savev2_adam_newdenselayer7_bias_v_read_readvariableop@
<savev2_adam_newdenselayeroutput_kernel_v_read_readvariableop>
:savev2_adam_newdenselayeroutput_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_newdenselayer1_kernel_read_readvariableop.savev2_newdenselayer1_bias_read_readvariableop0savev2_newdenselayer2_kernel_read_readvariableop.savev2_newdenselayer2_bias_read_readvariableop0savev2_newdenselayer3_kernel_read_readvariableop.savev2_newdenselayer3_bias_read_readvariableop0savev2_newdenselayer4_kernel_read_readvariableop.savev2_newdenselayer4_bias_read_readvariableop0savev2_newdenselayer5_kernel_read_readvariableop.savev2_newdenselayer5_bias_read_readvariableop0savev2_newdenselayer6_kernel_read_readvariableop.savev2_newdenselayer6_bias_read_readvariableop0savev2_newdenselayer7_kernel_read_readvariableop.savev2_newdenselayer7_bias_read_readvariableop5savev2_newdenselayeroutput_kernel_read_readvariableop3savev2_newdenselayeroutput_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_newdenselayer1_kernel_m_read_readvariableop5savev2_adam_newdenselayer1_bias_m_read_readvariableop7savev2_adam_newdenselayer2_kernel_m_read_readvariableop5savev2_adam_newdenselayer2_bias_m_read_readvariableop7savev2_adam_newdenselayer3_kernel_m_read_readvariableop5savev2_adam_newdenselayer3_bias_m_read_readvariableop7savev2_adam_newdenselayer4_kernel_m_read_readvariableop5savev2_adam_newdenselayer4_bias_m_read_readvariableop7savev2_adam_newdenselayer5_kernel_m_read_readvariableop5savev2_adam_newdenselayer5_bias_m_read_readvariableop7savev2_adam_newdenselayer6_kernel_m_read_readvariableop5savev2_adam_newdenselayer6_bias_m_read_readvariableop7savev2_adam_newdenselayer7_kernel_m_read_readvariableop5savev2_adam_newdenselayer7_bias_m_read_readvariableop<savev2_adam_newdenselayeroutput_kernel_m_read_readvariableop:savev2_adam_newdenselayeroutput_bias_m_read_readvariableop7savev2_adam_newdenselayer1_kernel_v_read_readvariableop5savev2_adam_newdenselayer1_bias_v_read_readvariableop7savev2_adam_newdenselayer2_kernel_v_read_readvariableop5savev2_adam_newdenselayer2_bias_v_read_readvariableop7savev2_adam_newdenselayer3_kernel_v_read_readvariableop5savev2_adam_newdenselayer3_bias_v_read_readvariableop7savev2_adam_newdenselayer4_kernel_v_read_readvariableop5savev2_adam_newdenselayer4_bias_v_read_readvariableop7savev2_adam_newdenselayer5_kernel_v_read_readvariableop5savev2_adam_newdenselayer5_bias_v_read_readvariableop7savev2_adam_newdenselayer6_kernel_v_read_readvariableop5savev2_adam_newdenselayer6_bias_v_read_readvariableop7savev2_adam_newdenselayer7_kernel_v_read_readvariableop5savev2_adam_newdenselayer7_bias_v_read_readvariableop<savev2_adam_newdenselayeroutput_kernel_v_read_readvariableop:savev2_adam_newdenselayeroutput_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :*@:@:	@?:?:
??:?:
??:?:
??:?:
??:?:	?@:@:@:: : : : : : : : : :*@:@:	@?:?:
??:?:
??:?:
??:?:
??:?:	?@:@:@::*@:@:	@?:?:
??:?:
??:?:
??:?:
??:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:*@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:*@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?@: '

_output_shapes
:@:$( 

_output_shapes

:@: )

_output_shapes
::$* 

_output_shapes

:*@: +

_output_shapes
:@:%,!

_output_shapes
:	@?:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:&2"
 
_output_shapes
:
??:!3

_output_shapes	
:?:&4"
 
_output_shapes
:
??:!5

_output_shapes	
:?:%6!

_output_shapes
:	?@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
:::

_output_shapes
: 
?
?
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_27199

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer1_layer_call_fn_27836

inputs
unknown:*@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_270802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????*: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_27206

inputs&
newdenselayer1_27081:*@"
newdenselayer1_27083:@'
newdenselayer2_27098:	@?#
newdenselayer2_27100:	?(
newdenselayer3_27115:
??#
newdenselayer3_27117:	?(
newdenselayer4_27132:
??#
newdenselayer4_27134:	?(
newdenselayer5_27149:
??#
newdenselayer5_27151:	?(
newdenselayer6_27166:
??#
newdenselayer6_27168:	?'
newdenselayer7_27183:	?@"
newdenselayer7_27185:@+
newdenselayeroutput_27200:@'
newdenselayeroutput_27202:
identity??&newDenseLayer1/StatefulPartitionedCall?&newDenseLayer2/StatefulPartitionedCall?&newDenseLayer3/StatefulPartitionedCall?&newDenseLayer4/StatefulPartitionedCall?&newDenseLayer5/StatefulPartitionedCall?&newDenseLayer6/StatefulPartitionedCall?&newDenseLayer7/StatefulPartitionedCall?+newDenseLayerOutput/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_270672
flatten_2/PartitionedCall?
&newDenseLayer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0newdenselayer1_27081newdenselayer1_27083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_270802(
&newDenseLayer1/StatefulPartitionedCall?
&newDenseLayer2/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer1/StatefulPartitionedCall:output:0newdenselayer2_27098newdenselayer2_27100*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_270972(
&newDenseLayer2/StatefulPartitionedCall?
&newDenseLayer3/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer2/StatefulPartitionedCall:output:0newdenselayer3_27115newdenselayer3_27117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_271142(
&newDenseLayer3/StatefulPartitionedCall?
&newDenseLayer4/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer3/StatefulPartitionedCall:output:0newdenselayer4_27132newdenselayer4_27134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_271312(
&newDenseLayer4/StatefulPartitionedCall?
&newDenseLayer5/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer4/StatefulPartitionedCall:output:0newdenselayer5_27149newdenselayer5_27151*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_271482(
&newDenseLayer5/StatefulPartitionedCall?
&newDenseLayer6/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer5/StatefulPartitionedCall:output:0newdenselayer6_27166newdenselayer6_27168*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_271652(
&newDenseLayer6/StatefulPartitionedCall?
&newDenseLayer7/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer6/StatefulPartitionedCall:output:0newdenselayer7_27183newdenselayer7_27185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_271822(
&newDenseLayer7/StatefulPartitionedCall?
+newDenseLayerOutput/StatefulPartitionedCallStatefulPartitionedCall/newDenseLayer7/StatefulPartitionedCall:output:0newdenselayeroutput_27200newdenselayeroutput_27202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_271992-
+newDenseLayerOutput/StatefulPartitionedCall?
IdentityIdentity4newDenseLayerOutput/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp'^newDenseLayer1/StatefulPartitionedCall'^newDenseLayer2/StatefulPartitionedCall'^newDenseLayer3/StatefulPartitionedCall'^newDenseLayer4/StatefulPartitionedCall'^newDenseLayer5/StatefulPartitionedCall'^newDenseLayer6/StatefulPartitionedCall'^newDenseLayer7/StatefulPartitionedCall,^newDenseLayerOutput/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 2P
&newDenseLayer1/StatefulPartitionedCall&newDenseLayer1/StatefulPartitionedCall2P
&newDenseLayer2/StatefulPartitionedCall&newDenseLayer2/StatefulPartitionedCall2P
&newDenseLayer3/StatefulPartitionedCall&newDenseLayer3/StatefulPartitionedCall2P
&newDenseLayer4/StatefulPartitionedCall&newDenseLayer4/StatefulPartitionedCall2P
&newDenseLayer5/StatefulPartitionedCall&newDenseLayer5/StatefulPartitionedCall2P
&newDenseLayer6/StatefulPartitionedCall&newDenseLayer6/StatefulPartitionedCall2P
&newDenseLayer7/StatefulPartitionedCall&newDenseLayer7/StatefulPartitionedCall2Z
+newDenseLayerOutput/StatefulPartitionedCall+newDenseLayerOutput/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_27131

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer4_layer_call_fn_27896

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_271312
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_27907

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_27080

inputs0
matmul_readvariableop_resource:*@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_27182

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_27867

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer2_layer_call_fn_27856

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_270972
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_27148

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_27618
flatten_2_input
unknown:*@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_270542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????
)
_user_specified_nameflatten_2_input
?
?
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_27987

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_27692

inputs
unknown:*@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_274112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_27847

inputs0
matmul_readvariableop_resource:*@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????*
 
_user_specified_nameinputs
?
?
.__inference_newDenseLayer6_layer_call_fn_27936

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_271652
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?%
!__inference__traced_restore_28362
file_prefix8
&assignvariableop_newdenselayer1_kernel:*@4
&assignvariableop_1_newdenselayer1_bias:@;
(assignvariableop_2_newdenselayer2_kernel:	@?5
&assignvariableop_3_newdenselayer2_bias:	?<
(assignvariableop_4_newdenselayer3_kernel:
??5
&assignvariableop_5_newdenselayer3_bias:	?<
(assignvariableop_6_newdenselayer4_kernel:
??5
&assignvariableop_7_newdenselayer4_bias:	?<
(assignvariableop_8_newdenselayer5_kernel:
??5
&assignvariableop_9_newdenselayer5_bias:	?=
)assignvariableop_10_newdenselayer6_kernel:
??6
'assignvariableop_11_newdenselayer6_bias:	?<
)assignvariableop_12_newdenselayer7_kernel:	?@5
'assignvariableop_13_newdenselayer7_bias:@@
.assignvariableop_14_newdenselayeroutput_kernel:@:
,assignvariableop_15_newdenselayeroutput_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: B
0assignvariableop_25_adam_newdenselayer1_kernel_m:*@<
.assignvariableop_26_adam_newdenselayer1_bias_m:@C
0assignvariableop_27_adam_newdenselayer2_kernel_m:	@?=
.assignvariableop_28_adam_newdenselayer2_bias_m:	?D
0assignvariableop_29_adam_newdenselayer3_kernel_m:
??=
.assignvariableop_30_adam_newdenselayer3_bias_m:	?D
0assignvariableop_31_adam_newdenselayer4_kernel_m:
??=
.assignvariableop_32_adam_newdenselayer4_bias_m:	?D
0assignvariableop_33_adam_newdenselayer5_kernel_m:
??=
.assignvariableop_34_adam_newdenselayer5_bias_m:	?D
0assignvariableop_35_adam_newdenselayer6_kernel_m:
??=
.assignvariableop_36_adam_newdenselayer6_bias_m:	?C
0assignvariableop_37_adam_newdenselayer7_kernel_m:	?@<
.assignvariableop_38_adam_newdenselayer7_bias_m:@G
5assignvariableop_39_adam_newdenselayeroutput_kernel_m:@A
3assignvariableop_40_adam_newdenselayeroutput_bias_m:B
0assignvariableop_41_adam_newdenselayer1_kernel_v:*@<
.assignvariableop_42_adam_newdenselayer1_bias_v:@C
0assignvariableop_43_adam_newdenselayer2_kernel_v:	@?=
.assignvariableop_44_adam_newdenselayer2_bias_v:	?D
0assignvariableop_45_adam_newdenselayer3_kernel_v:
??=
.assignvariableop_46_adam_newdenselayer3_bias_v:	?D
0assignvariableop_47_adam_newdenselayer4_kernel_v:
??=
.assignvariableop_48_adam_newdenselayer4_bias_v:	?D
0assignvariableop_49_adam_newdenselayer5_kernel_v:
??=
.assignvariableop_50_adam_newdenselayer5_bias_v:	?D
0assignvariableop_51_adam_newdenselayer6_kernel_v:
??=
.assignvariableop_52_adam_newdenselayer6_bias_v:	?C
0assignvariableop_53_adam_newdenselayer7_kernel_v:	?@<
.assignvariableop_54_adam_newdenselayer7_bias_v:@G
5assignvariableop_55_adam_newdenselayeroutput_kernel_v:@A
3assignvariableop_56_adam_newdenselayeroutput_bias_v:
identity_58??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp&assignvariableop_newdenselayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_newdenselayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_newdenselayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_newdenselayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_newdenselayer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_newdenselayer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_newdenselayer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_newdenselayer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_newdenselayer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_newdenselayer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_newdenselayer6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_newdenselayer6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_newdenselayer7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_newdenselayer7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_newdenselayeroutput_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_newdenselayeroutput_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_newdenselayer1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_newdenselayer1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_newdenselayer2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_newdenselayer2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_newdenselayer3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_newdenselayer3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_newdenselayer4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_newdenselayer4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_newdenselayer5_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_newdenselayer5_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_newdenselayer6_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_newdenselayer6_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_newdenselayer7_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_newdenselayer7_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_newdenselayeroutput_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_newdenselayeroutput_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_newdenselayer1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_newdenselayer1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp0assignvariableop_43_adam_newdenselayer2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp.assignvariableop_44_adam_newdenselayer2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp0assignvariableop_45_adam_newdenselayer3_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_adam_newdenselayer3_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp0assignvariableop_47_adam_newdenselayer4_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp.assignvariableop_48_adam_newdenselayer4_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp0assignvariableop_49_adam_newdenselayer5_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp.assignvariableop_50_adam_newdenselayer5_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp0assignvariableop_51_adam_newdenselayer6_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_newdenselayer6_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp0assignvariableop_53_adam_newdenselayer7_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_newdenselayer7_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_newdenselayeroutput_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_newdenselayeroutput_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_569
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57f
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_58?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_58Identity_58:output:0*?
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
flatten_2_input<
!serving_default_flatten_2_input:0?????????G
newDenseLayerOutput0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4bias
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Eiter

Fbeta_1

Gbeta_2
	Hdecay
Ilearning_ratem?m?m?m?!m?"m?'m?(m?-m?.m?3m?4m?9m?:m??m?@m?v?v?v?v?!v?"v?'v?(v?-v?.v?3v?4v?9v?:v??v?@v?"
	optimizer
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15"
trackable_list_wrapper
?
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
?14
@15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Jlayers
Kmetrics
Lnon_trainable_variables
	variables
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Olayers
Pmetrics
Qnon_trainable_variables
	variables
regularization_losses
Rlayer_regularization_losses
Slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%*@2newDenseLayer1/kernel
!:@2newDenseLayer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Tlayers
Umetrics
Vnon_trainable_variables
	variables
regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	@?2newDenseLayer2/kernel
": ?2newDenseLayer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables

Ylayers
Zmetrics
[non_trainable_variables
	variables
regularization_losses
\layer_regularization_losses
]layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2newDenseLayer3/kernel
": ?2newDenseLayer3/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#trainable_variables

^layers
_metrics
`non_trainable_variables
$	variables
%regularization_losses
alayer_regularization_losses
blayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2newDenseLayer4/kernel
": ?2newDenseLayer4/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables

clayers
dmetrics
enon_trainable_variables
*	variables
+regularization_losses
flayer_regularization_losses
glayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2newDenseLayer5/kernel
": ?2newDenseLayer5/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/trainable_variables

hlayers
imetrics
jnon_trainable_variables
0	variables
1regularization_losses
klayer_regularization_losses
llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2newDenseLayer6/kernel
": ?2newDenseLayer6/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5trainable_variables

mlayers
nmetrics
onon_trainable_variables
6	variables
7regularization_losses
player_regularization_losses
qlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	?@2newDenseLayer7/kernel
!:@2newDenseLayer7/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;trainable_variables

rlayers
smetrics
tnon_trainable_variables
<	variables
=regularization_losses
ulayer_regularization_losses
vlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*@2newDenseLayerOutput/kernel
&:$2newDenseLayerOutput/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Atrainable_variables

wlayers
xmetrics
ynon_trainable_variables
B	variables
Cregularization_losses
zlayer_regularization_losses
{layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
|0
}1"
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
P
	~total
	count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:**@2Adam/newDenseLayer1/kernel/m
&:$@2Adam/newDenseLayer1/bias/m
-:+	@?2Adam/newDenseLayer2/kernel/m
':%?2Adam/newDenseLayer2/bias/m
.:,
??2Adam/newDenseLayer3/kernel/m
':%?2Adam/newDenseLayer3/bias/m
.:,
??2Adam/newDenseLayer4/kernel/m
':%?2Adam/newDenseLayer4/bias/m
.:,
??2Adam/newDenseLayer5/kernel/m
':%?2Adam/newDenseLayer5/bias/m
.:,
??2Adam/newDenseLayer6/kernel/m
':%?2Adam/newDenseLayer6/bias/m
-:+	?@2Adam/newDenseLayer7/kernel/m
&:$@2Adam/newDenseLayer7/bias/m
1:/@2!Adam/newDenseLayerOutput/kernel/m
+:)2Adam/newDenseLayerOutput/bias/m
,:**@2Adam/newDenseLayer1/kernel/v
&:$@2Adam/newDenseLayer1/bias/v
-:+	@?2Adam/newDenseLayer2/kernel/v
':%?2Adam/newDenseLayer2/bias/v
.:,
??2Adam/newDenseLayer3/kernel/v
':%?2Adam/newDenseLayer3/bias/v
.:,
??2Adam/newDenseLayer4/kernel/v
':%?2Adam/newDenseLayer4/bias/v
.:,
??2Adam/newDenseLayer5/kernel/v
':%?2Adam/newDenseLayer5/bias/v
.:,
??2Adam/newDenseLayer6/kernel/v
':%?2Adam/newDenseLayer6/bias/v
-:+	?@2Adam/newDenseLayer7/kernel/v
&:$@2Adam/newDenseLayer7/bias/v
1:/@2!Adam/newDenseLayerOutput/kernel/v
+:)2Adam/newDenseLayerOutput/bias/v
?2?
*__inference_sequential_layer_call_fn_27241
*__inference_sequential_layer_call_fn_27655
*__inference_sequential_layer_call_fn_27692
*__inference_sequential_layer_call_fn_27483?
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
?B?
 __inference__wrapped_model_27054flatten_2_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_27754
E__inference_sequential_layer_call_and_return_conditional_losses_27816
E__inference_sequential_layer_call_and_return_conditional_losses_27528
E__inference_sequential_layer_call_and_return_conditional_losses_27573?
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
?2?
)__inference_flatten_2_layer_call_fn_27821?
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
D__inference_flatten_2_layer_call_and_return_conditional_losses_27827?
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
.__inference_newDenseLayer1_layer_call_fn_27836?
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
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_27847?
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
.__inference_newDenseLayer2_layer_call_fn_27856?
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
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_27867?
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
.__inference_newDenseLayer3_layer_call_fn_27876?
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
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_27887?
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
.__inference_newDenseLayer4_layer_call_fn_27896?
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
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_27907?
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
.__inference_newDenseLayer5_layer_call_fn_27916?
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
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_27927?
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
.__inference_newDenseLayer6_layer_call_fn_27936?
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
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_27947?
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
.__inference_newDenseLayer7_layer_call_fn_27956?
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
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_27967?
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
3__inference_newDenseLayerOutput_layer_call_fn_27976?
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
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_27987?
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
?B?
#__inference_signature_wrapper_27618flatten_2_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_27054?!"'(-.349:?@<?9
2?/
-?*
flatten_2_input?????????
? "I?F
D
newDenseLayerOutput-?*
newDenseLayerOutput??????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_27827\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????*
? |
)__inference_flatten_2_layer_call_fn_27821O3?0
)?&
$?!
inputs?????????
? "??????????*?
I__inference_newDenseLayer1_layer_call_and_return_conditional_losses_27847\/?,
%?"
 ?
inputs?????????*
? "%?"
?
0?????????@
? ?
.__inference_newDenseLayer1_layer_call_fn_27836O/?,
%?"
 ?
inputs?????????*
? "??????????@?
I__inference_newDenseLayer2_layer_call_and_return_conditional_losses_27867]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? ?
.__inference_newDenseLayer2_layer_call_fn_27856P/?,
%?"
 ?
inputs?????????@
? "????????????
I__inference_newDenseLayer3_layer_call_and_return_conditional_losses_27887^!"0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_newDenseLayer3_layer_call_fn_27876Q!"0?-
&?#
!?
inputs??????????
? "????????????
I__inference_newDenseLayer4_layer_call_and_return_conditional_losses_27907^'(0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_newDenseLayer4_layer_call_fn_27896Q'(0?-
&?#
!?
inputs??????????
? "????????????
I__inference_newDenseLayer5_layer_call_and_return_conditional_losses_27927^-.0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_newDenseLayer5_layer_call_fn_27916Q-.0?-
&?#
!?
inputs??????????
? "????????????
I__inference_newDenseLayer6_layer_call_and_return_conditional_losses_27947^340?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_newDenseLayer6_layer_call_fn_27936Q340?-
&?#
!?
inputs??????????
? "????????????
I__inference_newDenseLayer7_layer_call_and_return_conditional_losses_27967]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ?
.__inference_newDenseLayer7_layer_call_fn_27956P9:0?-
&?#
!?
inputs??????????
? "??????????@?
N__inference_newDenseLayerOutput_layer_call_and_return_conditional_losses_27987\?@/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
3__inference_newDenseLayerOutput_layer_call_fn_27976O?@/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_sequential_layer_call_and_return_conditional_losses_27528!"'(-.349:?@D?A
:?7
-?*
flatten_2_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_27573!"'(-.349:?@D?A
:?7
-?*
flatten_2_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_27754v!"'(-.349:?@;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_27816v!"'(-.349:?@;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_27241r!"'(-.349:?@D?A
:?7
-?*
flatten_2_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_27483r!"'(-.349:?@D?A
:?7
-?*
flatten_2_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_27655i!"'(-.349:?@;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_27692i!"'(-.349:?@;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_27618?!"'(-.349:?@O?L
? 
E?B
@
flatten_2_input-?*
flatten_2_input?????????"I?F
D
newDenseLayerOutput-?*
newDenseLayerOutput?????????