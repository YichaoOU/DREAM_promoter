??<
?*?*
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
-
Lgamma
x"T
y"T"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
1
Sign
x"T
y"T"
Ttype:
2
	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
5
Xlogy
x"T
y"T
z"T"
Ttype:	
2
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??:
?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*[
shared_nameLJenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w
?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w/Read/ReadVariableOpReadVariableOpJenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w*
_output_shapes
:	?*
dtype0
?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*a
shared_nameRPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/b
?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/b/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/b*
_output_shapes	
:?*
dtype0
?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*a
shared_nameRPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/w
?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/w/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w
?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w
?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w
?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w* 
_output_shapes
:
??*
dtype0
?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*[
shared_nameLJenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w
?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w/Read/ReadVariableOpReadVariableOpJenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w*
_output_shapes
:	?*
dtype0
?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*a
shared_nameRPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/b
?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/b/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/b*
_output_shapes	
:?*
dtype0
?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*a
shared_nameRPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/w
?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/w/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w
?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w
?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w
?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w* 
_output_shapes
:
??*
dtype0
?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*[
shared_nameLJenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w
?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w/Read/ReadVariableOpReadVariableOpJenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w*
_output_shapes
:	?*
dtype0
?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*a
shared_nameRPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/b
?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/b/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/b*
_output_shapes	
:?*
dtype0
?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*a
shared_nameRPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/w
?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/w/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w
?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w
?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w
?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w* 
_output_shapes
:
??*
dtype0
?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*[
shared_nameLJenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w
?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w/Read/ReadVariableOpReadVariableOpJenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w*
_output_shapes
:	?*
dtype0
?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*a
shared_nameRPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/b
?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/b/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/b*
_output_shapes	
:?*
dtype0
?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*a
shared_nameRPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/w
?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/w/Read/ReadVariableOpReadVariableOpPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w
?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w
?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w* 
_output_shapes
:
??*
dtype0
?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*Y
shared_nameJHenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w
?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w/Read/ReadVariableOpReadVariableOpHenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w* 
_output_shapes
:
??*
dtype0
?
denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average
?
xenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average*#
_output_shapes
:?*
dtype0
?
cenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*t
shared_nameecenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden
?
wenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpcenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden*#
_output_shapes
:?*
dtype0
?
denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter
?
xenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
fenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1
?
zenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1*#
_output_shapes
:?*
dtype0
?
eenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*v
shared_namegeenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1
?
yenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpeenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1*#
_output_shapes
:?*
dtype0
?
fenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1
?
zenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average
?
xenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average*#
_output_shapes
:?*
dtype0
?
cenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*t
shared_nameecenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden
?
wenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpcenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden*#
_output_shapes
:?*
dtype0
?
denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter
?
xenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
fenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1
?
zenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1*#
_output_shapes
:?*
dtype0
?
eenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*v
shared_namegeenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1
?
yenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpeenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1*#
_output_shapes
:?*
dtype0
?
fenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1
?
zenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average
?
xenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average*"
_output_shapes
:@*
dtype0
?
cenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*t
shared_nameecenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden
?
wenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpcenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden*"
_output_shapes
:@*
dtype0
?
denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter
?
xenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
fenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1
?
zenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1*"
_output_shapes
:@*
dtype0
?
eenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*v
shared_namegeenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1
?
yenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpeenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:@*
dtype0
?
fenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1
?
zenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average
?
xenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average*"
_output_shapes
:@*
dtype0
?
cenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*t
shared_nameecenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden
?
wenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpcenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden*"
_output_shapes
:@*
dtype0
?
denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *u
shared_namefdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter
?
xenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
fenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1
?
zenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1*"
_output_shapes
:@*
dtype0
?
eenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*v
shared_namegeenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1
?
yenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpeenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:@*
dtype0
?
fenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *w
shared_namehfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1
?
zenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
;enformer/trunk/transformer/transformer_block_3/mlp/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;enformer/trunk/transformer/transformer_block_3/mlp/linear/b
?
Oenformer/trunk/transformer/transformer_block_3/mlp/linear/b/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_3/mlp/linear/b*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_3/mlp/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*L
shared_name=;enformer/trunk/transformer/transformer_block_3/mlp/linear/w
?
Oenformer/trunk/transformer/transformer_block_3/mlp/linear/w/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_3/mlp/linear/w* 
_output_shapes
:
??*
dtype0
?
=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1
?
Qenformer/trunk/transformer/transformer_block_3/mlp/linear/b_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1*
_output_shapes	
:?*
dtype0
?
=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*N
shared_name?=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1
?
Qenformer/trunk/transformer/transformer_block_3/mlp/linear/w_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1* 
_output_shapes
:
??*
dtype0
?
Denformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias
?
[enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias*&
_output_shapes
:@*
dtype0
?
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias
?
[enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias*&
_output_shapes
:@*
dtype0
?
Denformer/trunk/transformer/transformer_block_3/mha/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_2/mlp/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;enformer/trunk/transformer/transformer_block_2/mlp/linear/b
?
Oenformer/trunk/transformer/transformer_block_2/mlp/linear/b/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_2/mlp/linear/b*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_2/mlp/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*L
shared_name=;enformer/trunk/transformer/transformer_block_2/mlp/linear/w
?
Oenformer/trunk/transformer/transformer_block_2/mlp/linear/w/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_2/mlp/linear/w* 
_output_shapes
:
??*
dtype0
?
=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1
?
Qenformer/trunk/transformer/transformer_block_2/mlp/linear/b_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1*
_output_shapes	
:?*
dtype0
?
=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*N
shared_name?=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1
?
Qenformer/trunk/transformer/transformer_block_2/mlp/linear/w_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1* 
_output_shapes
:
??*
dtype0
?
Denformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias
?
[enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias*&
_output_shapes
:@*
dtype0
?
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias
?
[enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias*&
_output_shapes
:@*
dtype0
?
Denformer/trunk/transformer/transformer_block_2/mha/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_1/mlp/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;enformer/trunk/transformer/transformer_block_1/mlp/linear/b
?
Oenformer/trunk/transformer/transformer_block_1/mlp/linear/b/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_1/mlp/linear/b*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_1/mlp/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*L
shared_name=;enformer/trunk/transformer/transformer_block_1/mlp/linear/w
?
Oenformer/trunk/transformer/transformer_block_1/mlp/linear/w/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_1/mlp/linear/w* 
_output_shapes
:
??*
dtype0
?
=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1
?
Qenformer/trunk/transformer/transformer_block_1/mlp/linear/b_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1*
_output_shapes	
:?*
dtype0
?
=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*N
shared_name?=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1
?
Qenformer/trunk/transformer/transformer_block_1/mlp/linear/w_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1* 
_output_shapes
:
??*
dtype0
?
Denformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias
?
[enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias*&
_output_shapes
:@*
dtype0
?
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias
?
[enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias*&
_output_shapes
:@*
dtype0
?
Denformer/trunk/transformer/transformer_block_1/mha/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_0/mlp/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*L
shared_name=;enformer/trunk/transformer/transformer_block_0/mlp/linear/b
?
Oenformer/trunk/transformer/transformer_block_0/mlp/linear/b/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_0/mlp/linear/b*
_output_shapes	
:?*
dtype0
?
;enformer/trunk/transformer/transformer_block_0/mlp/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*L
shared_name=;enformer/trunk/transformer/transformer_block_0/mlp/linear/w
?
Oenformer/trunk/transformer/transformer_block_0/mlp/linear/w/Read/ReadVariableOpReadVariableOp;enformer/trunk/transformer/transformer_block_0/mlp/linear/w* 
_output_shapes
:
??*
dtype0
?
=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*N
shared_name?=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1
?
Qenformer/trunk/transformer/transformer_block_0/mlp/linear/b_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1*
_output_shapes	
:?*
dtype0
?
=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*N
shared_name?=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1
?
Qenformer/trunk/transformer/transformer_block_0/mlp/linear/w_1/Read/ReadVariableOpReadVariableOp=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1* 
_output_shapes
:
??*
dtype0
?
Denformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias
?
[enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias*&
_output_shapes
:@*
dtype0
?
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*X
shared_nameIGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias
?
[enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias/Read/ReadVariableOpReadVariableOpGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias*&
_output_shapes
:@*
dtype0
?
Denformer/trunk/transformer/transformer_block_0/mha/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*U
shared_nameFDenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offset
?
Xenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offset/Read/ReadVariableOpReadVariableOpDenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offset*
_output_shapes	
:?*
dtype0
?
Cenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*T
shared_nameECenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scale
?
Wenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scale/Read/ReadVariableOpReadVariableOpCenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scale*
_output_shapes	
:?*
dtype0
?
Kenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/b
?
_enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/b*
_output_shapes	
:?*
dtype0
?
Kenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/w
?
_enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/w*$
_output_shapes
:??*
dtype0
?
aenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*r
shared_namecaenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offset
?
uenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpaenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offset*
_output_shapes	
:?*
dtype0
?
`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*q
shared_nameb`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scale
?
tenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scale*
_output_shapes	
:?*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average
?
nenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average*#
_output_shapes
:?*
dtype0
?
Yenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*j
shared_name[Yenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden
?
menformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpYenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden*#
_output_shapes
:?*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter
?
nenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1
?
penformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1*#
_output_shapes
:?*
dtype0
?
[enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*l
shared_name][enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1
?
oenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOp[enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1*#
_output_shapes
:?*
dtype0
?
\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1
?
penformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
Kenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b
?
_enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b*
_output_shapes	
:?*
dtype0
?
Kenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/w
?
_enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/w*$
_output_shapes
:??*
dtype0
?
aenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*r
shared_namecaenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offset
?
uenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpaenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offset*
_output_shapes	
:?*
dtype0
?
`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*q
shared_nameb`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scale
?
tenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scale*
_output_shapes	
:?*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average
?
nenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average*"
_output_shapes
:@*
dtype0
?
Yenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*j
shared_name[Yenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden
?
menformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpYenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden*"
_output_shapes
:@*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter
?
nenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1
?
penformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1*"
_output_shapes
:@*
dtype0
?
[enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*l
shared_name][enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1
?
oenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOp[enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:@*
dtype0
?
\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1
?
penformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
Kenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b
?
_enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b*
_output_shapes
:@*
dtype0
?
Kenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/w
?
_enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/w*"
_output_shapes
:@@*
dtype0
?
aenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*r
shared_namecaenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offset
?
uenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpaenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offset*
_output_shapes
:@*
dtype0
?
`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*q
shared_nameb`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scale
?
tenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scale*
_output_shapes
:@*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average
?
nenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average*"
_output_shapes
:@*
dtype0
?
Yenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*j
shared_name[Yenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden
?
menformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpYenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden*"
_output_shapes
:@*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter
?
nenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1
?
penformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1*"
_output_shapes
:@*
dtype0
?
[enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*l
shared_name][enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1
?
oenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOp[enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:@*
dtype0
?
\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1
?
penformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
Kenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b
?
_enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b*
_output_shapes
:@*
dtype0
?
Kenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*\
shared_nameMKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/w
?
_enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/w*"
_output_shapes
:@@*
dtype0
?
aenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*r
shared_namecaenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offset
?
uenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpaenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offset*
_output_shapes
:@*
dtype0
?
`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*q
shared_nameb`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scale
?
tenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scale*
_output_shapes
:@*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average
?
nenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average*"
_output_shapes
:0*
dtype0
?
Yenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*j
shared_name[Yenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden
?
menformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpYenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden*"
_output_shapes
:0*
dtype0
?
Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *k
shared_name\Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter
?
nenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpZenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1
?
penformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1*"
_output_shapes
:0*
dtype0
?
[enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*l
shared_name][enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1
?
oenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOp[enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:0*
dtype0
?
\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *m
shared_name^\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1
?
penformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOp\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
Aenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b
?
Uenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b*
_output_shapes	
:?*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/w
?
Uenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/w*$
_output_shapes
:??*
dtype0
?
Wenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*h
shared_nameYWenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offset
?
kenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpWenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offset*
_output_shapes	
:?*
dtype0
?
Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*g
shared_nameXVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scale
?
jenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scale*
_output_shapes	
:?*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/b
?
Uenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/b*
_output_shapes	
:?*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/w
?
Uenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/w*#
_output_shapes
:@?*
dtype0
?
Wenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*h
shared_nameYWenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offset
?
kenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpWenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offset*
_output_shapes
:@*
dtype0
?
Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*g
shared_nameXVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scale
?
jenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scale*
_output_shapes
:@*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/b
?
Uenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/b*
_output_shapes
:@*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/w
?
Uenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/w*"
_output_shapes
:@@*
dtype0
?
Wenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*h
shared_nameYWenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offset
?
kenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpWenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offset*
_output_shapes
:@*
dtype0
?
Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*g
shared_nameXVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scale
?
jenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scale*
_output_shapes
:@*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/b
?
Uenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/b*
_output_shapes
:@*
dtype0
?
Aenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*R
shared_nameCAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/w
?
Uenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOpAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/w*"
_output_shapes
:0@*
dtype0
?
Wenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*h
shared_nameYWenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offset
?
kenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpWenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offset*
_output_shapes
:0*
dtype0
?
Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*g
shared_nameXVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scale
?
jenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scale*
_output_shapes
:0*
dtype0
?
Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*\
shared_nameMKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average
?
_enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average*"
_output_shapes
:0*
dtype0
?
Jenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*[
shared_nameLJenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden
?
^enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpJenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden*"
_output_shapes
:0*
dtype0
?
Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *\
shared_nameMKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter
?
_enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
Menformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*^
shared_nameOMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1
?
aenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1*"
_output_shapes
:0*
dtype0
?
Lenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*]
shared_nameNLenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1
?
`enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpLenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1*"
_output_shapes
:0*
dtype0
?
Menformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *^
shared_nameOMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1
?
aenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/averageVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*]
shared_nameNLenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average
?
`enformer/trunk/final_pointwise/conv_block/exponential_moving_average/average/Read/ReadVariableOpReadVariableOpLenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average*#
_output_shapes
:?*
dtype0
?
Kenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hiddenVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*\
shared_nameMKenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden
?
_enformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpReadVariableOpKenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden*#
_output_shapes
:?*
dtype0
?
Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *]
shared_nameNLenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter
?
`enformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter/Read/ReadVariableOpReadVariableOpLenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter*
_output_shapes
: *
dtype0	
?
Nenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*_
shared_namePNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1
?
benformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpReadVariableOpNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1*#
_output_shapes
:?*
dtype0
?
Menformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*^
shared_nameOMenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1
?
aenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpReadVariableOpMenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1*#
_output_shapes
:?*
dtype0
?
Nenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *_
shared_namePNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1
?
benformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpReadVariableOpNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1*
_output_shapes
: *
dtype0	
?
2enformer/trunk/stem/pointwise_conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*C
shared_name42enformer/trunk/stem/pointwise_conv_block/conv1_d/b
?
Fenformer/trunk/stem/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOp2enformer/trunk/stem/pointwise_conv_block/conv1_d/b*
_output_shapes
:0*
dtype0
?
2enformer/trunk/stem/pointwise_conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*C
shared_name42enformer/trunk/stem/pointwise_conv_block/conv1_d/w
?
Fenformer/trunk/stem/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOp2enformer/trunk/stem/pointwise_conv_block/conv1_d/w*"
_output_shapes
:00*
dtype0
?
Henformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*Y
shared_nameJHenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset
?
\enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpHenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset*
_output_shapes
:0*
dtype0
?
Genformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*X
shared_nameIGenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scale
?
[enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpGenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scale*
_output_shapes
:0*
dtype0
?
3enformer/trunk/final_pointwise/conv_block/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53enformer/trunk/final_pointwise/conv_block/conv1_d/b
?
Genformer/trunk/final_pointwise/conv_block/conv1_d/b/Read/ReadVariableOpReadVariableOp3enformer/trunk/final_pointwise/conv_block/conv1_d/b*
_output_shapes	
:?*
dtype0
?
3enformer/trunk/final_pointwise/conv_block/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*D
shared_name53enformer/trunk/final_pointwise/conv_block/conv1_d/w
?
Genformer/trunk/final_pointwise/conv_block/conv1_d/w/Read/ReadVariableOpReadVariableOp3enformer/trunk/final_pointwise/conv_block/conv1_d/w*$
_output_shapes
:??*
dtype0
?
Ienformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset
?
]enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpReadVariableOpIenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset*
_output_shapes	
:?*
dtype0
?
Henformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Y
shared_nameJHenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scale
?
\enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpReadVariableOpHenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scale*
_output_shapes	
:?*
dtype0
?
enformer/trunk/stem/conv1_d/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameenformer/trunk/stem/conv1_d/b
?
1enformer/trunk/stem/conv1_d/b/Read/ReadVariableOpReadVariableOpenformer/trunk/stem/conv1_d/b*
_output_shapes
:0*
dtype0
?
enformer/trunk/stem/conv1_d/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameenformer/trunk/stem/conv1_d/w
?
1enformer/trunk/stem/conv1_d/w/Read/ReadVariableOpReadVariableOpenformer/trunk/stem/conv1_d/w*"
_output_shapes
:0*
dtype0
?
"enformer/heads/head_yeast/output/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"enformer/heads/head_yeast/output/b
?
6enformer/heads/head_yeast/output/b/Read/ReadVariableOpReadVariableOp"enformer/heads/head_yeast/output/b*
_output_shapes
:*
dtype0
?
"enformer/heads/head_yeast/output/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"enformer/heads/head_yeast/output/w
?
6enformer/heads/head_yeast/output/w/Read/ReadVariableOpReadVariableOp"enformer/heads/head_yeast/output/w*
_output_shapes
:	?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
B

_trunk

_heads
predict_on_batch

signatures*

_layers*

	yeast*

trace_0* 

serving_default* 
'
	0

1
2
3
4*

_layers*
* 
* 

_layers*

_layers*

_layers*
* 

_layers*

0
1*

0
1
2*
 
0
1
2
3*
 
0
1
2
3*

 0
!1
"2*
* 

#w
$b*

%w
&b*

'_module*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 

._layers*

/_layers*

0_layers*

1_layers*

2_layers*

3_layers*

4_layers*

5_layers*

6_layers*
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
* 
oi
VARIABLE_VALUE"enformer/heads/head_yeast/output/w3_heads/yeast/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE"enformer/heads/head_yeast/output/b3_heads/yeast/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEenformer/trunk/stem/conv1_d/w7_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEenformer/trunk/stem/conv1_d/b7_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUE*

=_layers*
* 
* 
* 
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

Ctrace_0* 

Dtrace_0* 

E0
F1
G2*

H0
I1
J2*

K0
L1
M2*

N0
O1
P2*

Q0
R1*

S0
T1*

U0
V1*

W0
X1*

Y0
Z2*
* 
* 
* 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

`trace_0* 

atrace_0* 

b0
c2*
* 
* 
* 
* 
* 
* 
* 

d_layers*

e_module*
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 

l_layers*

m_module*
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 

t_layers*

u_module*
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|_layers*

}_module*
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?_module*

?_module*

?_module*

?_module*

?_module*

?_module*

?_module*

?_module*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*
* 
* 
* 
* 
* 
* 
* 
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*

?0
?2*

?_layers*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?2*

?_layers*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?2*

?_layers*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?2*

?_layers*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?_layers*

?_layers*

?_layers*

?_layers*

?_layers*

?_layers*

?_layers*

?_layers*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEHenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scaleE_trunk/_layers/4/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEIenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offsetF_trunk/_layers/4/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3enformer/trunk/final_pointwise/conv_block/conv1_d/wA_trunk/_layers/4/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3enformer/trunk/final_pointwise/conv_block/conv1_d/bA_trunk/_layers/4/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEGenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scaleM_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offsetN_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2enformer/trunk/stem/pointwise_conv_block/conv1_d/wI_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2enformer/trunk/stem/pointwise_conv_block/conv1_d/bI_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*

?0
?2*
* 
* 
* 
* 
* 
* 
* 
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*

?0
?2*
* 
* 
* 
* 
* 
* 
* 
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*

?0
?2*
* 
* 
* 
* 
* 
* 
* 
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*

?0
?2*
* 
* 
* 
* 
* 
* 
* 

?0
?1
?2*
,
?0
?1
?2
?4
?5*

?0
?1
?2*
,
?0
?1
?2
?4
?5*

?0
?1
?2*
,
?0
?1
?2
?4
?5*

?0
?1
?2*
,
?0
?1
?2
?4
?5*
??
VARIABLE_VALUENenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1T_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1S_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUENenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1S_trunk/_layers/4/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUELenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counterX_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hiddenW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUELenformer/trunk/final_pointwise/conv_block/exponential_moving_average/averageW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1\_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUELenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter`_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scaleO_trunk/_layers/1/_layers/0/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEWenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offsetP_trunk/_layers/1/_layers/0/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/wK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/bK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scaleO_trunk/_layers/1/_layers/1/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEWenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offsetP_trunk/_layers/1/_layers/1/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/wK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/bK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scaleO_trunk/_layers/1/_layers/2/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEWenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offsetP_trunk/_layers/1/_layers/2/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/wK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/bK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUEVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scaleO_trunk/_layers/1/_layers/3/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEWenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offsetP_trunk/_layers/1/_layers/3/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/wK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/bK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
E
?moving_mean
?moving_variance

?scale
?offset*

?w
?b*


?scale
?offset*
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias*
* 


?scale
?offset*

?w
?b*
* 

?w
?b*
* 


?scale
?offset*
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias*
* 


?scale
?offset*

?w
?b*
* 

?w
?b*
* 


?scale
?offset*
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias*
* 


?scale
?offset*

?w
?b*
* 

?w
?b*
* 


?scale
?offset*
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias*
* 


?scale
?offset*

?w
?b*
* 

?w
?b*
* 
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1^_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE[enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counterb_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEYenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hiddena_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/averagea_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUE`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scaleW_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEaenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offsetX_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/wS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/bS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1^_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE[enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counterb_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEYenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hiddena_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/averagea_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUE`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scaleW_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEaenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offsetX_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/wS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/bS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1^_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE[enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counterb_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEYenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hiddena_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/averagea_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUE`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scaleW_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEaenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offsetX_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/wS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/bS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1^_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE[enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counterb_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEYenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hiddena_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEZenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/averagea_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
/
?_counter
?_hidden
?average*
/
?_counter
?_hidden
?average*
??
VARIABLE_VALUE`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scaleW_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEaenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offsetX_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/wS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/bS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scaleW_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offsetX_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
* 

?w*

?w*

?w*

?w
?b*

?w*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scaleW_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offsetX_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1S_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1S_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_0/mlp/linear/wS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_0/mlp/linear/bS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scaleW_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offsetX_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
* 

?w*

?w*

?w*

?w
?b*

?w*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scaleW_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offsetX_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1S_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1S_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_1/mlp/linear/wS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_1/mlp/linear/bS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scaleW_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offsetX_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
* 

?w*

?w*

?w*

?w
?b*

?w*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scaleW_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offsetX_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1S_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1S_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_2/mlp/linear/wS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_2/mlp/linear/bS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scaleW_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offsetX_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
* 

?w*

?w*

?w*

?w
?b*

?w*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scaleW_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offsetX_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1S_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1S_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_3/mlp/linear/wS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;enformer/trunk/transformer/transformer_block_3/mlp/linear/bS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1f_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1e_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1e_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counterj_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hiddeni_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/averagei_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1f_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1e_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1e_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counterj_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hiddeni_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/averagei_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1f_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1e_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1e_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counterj_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hiddeni_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/averagei_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1f_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEeenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1e_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1e_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counterj_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEcenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hiddeni_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEdenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/averagei_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/wd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/bd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w^_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/wd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/bd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w^_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/wd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/bd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w^_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/wd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/bd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w^_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_args_0Placeholder*-
_output_shapes
:???????????*
dtype0*"
shape:???????????
?P
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0enformer/trunk/stem/conv1_d/wenformer/trunk/stem/conv1_d/bMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/averageGenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scaleHenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset2enformer/trunk/stem/pointwise_conv_block/conv1_d/w2enformer/trunk/stem/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/averageVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/bfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/averageVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/bfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/averageVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/bfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/averageVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/bfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/bCenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offsetHenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/wHenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/wHenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/wJenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/wGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_biasGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_biasPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/wPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/bCenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_0/mlp/linear/w;enformer/trunk/transformer/transformer_block_0/mlp/linear/bCenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offsetHenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/wHenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/wHenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/wJenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/wGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_biasGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_biasPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/wPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/bCenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_1/mlp/linear/w;enformer/trunk/transformer/transformer_block_1/mlp/linear/bCenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offsetHenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/wHenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/wHenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/wJenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/wGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_biasGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_biasPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/wPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/bCenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_2/mlp/linear/w;enformer/trunk/transformer/transformer_block_2/mlp/linear/bCenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offsetHenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/wHenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/wHenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/wJenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/wGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_biasGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_biasPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/wPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/bCenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_3/mlp/linear/w;enformer/trunk/transformer/transformer_block_3/mlp/linear/bNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/averageHenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scaleIenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset3enformer/trunk/final_pointwise/conv_block/conv1_d/w3enformer/trunk/final_pointwise/conv_block/conv1_d/b"enformer/heads/head_yeast/output/w"enformer/heads/head_yeast/output/b*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_4768291
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
??
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6enformer/heads/head_yeast/output/w/Read/ReadVariableOp6enformer/heads/head_yeast/output/b/Read/ReadVariableOp1enformer/trunk/stem/conv1_d/w/Read/ReadVariableOp1enformer/trunk/stem/conv1_d/b/Read/ReadVariableOp\enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOp]enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpGenformer/trunk/final_pointwise/conv_block/conv1_d/w/Read/ReadVariableOpGenformer/trunk/final_pointwise/conv_block/conv1_d/b/Read/ReadVariableOp[enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOp\enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpFenformer/trunk/stem/pointwise_conv_block/conv1_d/w/Read/ReadVariableOpFenformer/trunk/stem/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpbenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpaenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpbenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1/Read/ReadVariableOp`enformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter/Read/ReadVariableOp_enformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden/Read/ReadVariableOp`enformer/trunk/final_pointwise/conv_block/exponential_moving_average/average/Read/ReadVariableOpaenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOp`enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpaenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOp_enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOp^enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOp_enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpjenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpkenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/w/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/b/Read/ReadVariableOpjenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpkenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/w/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/b/Read/ReadVariableOpjenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpkenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/w/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/b/Read/ReadVariableOpjenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpkenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/w/Read/ReadVariableOpUenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpoenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter/Read/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average/Read/ReadVariableOptenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpuenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/w/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpoenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter/Read/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average/Read/ReadVariableOptenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpuenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/w/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpoenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter/Read/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average/Read/ReadVariableOptenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpuenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/w/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpoenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter/Read/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden/Read/ReadVariableOpnenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average/Read/ReadVariableOptenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scale/Read/ReadVariableOpuenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offset/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/w/Read/ReadVariableOp_enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/b/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offset/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_0/mlp/linear/w_1/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_0/mlp/linear/b_1/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_0/mlp/linear/w/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_0/mlp/linear/b/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offset/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_1/mlp/linear/w_1/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_1/mlp/linear/b_1/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_1/mlp/linear/w/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_1/mlp/linear/b/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offset/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_2/mlp/linear/w_1/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_2/mlp/linear/b_1/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_2/mlp/linear/w/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_2/mlp/linear/b/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offset/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias/Read/ReadVariableOp[enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias/Read/ReadVariableOpWenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scale/Read/ReadVariableOpXenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_3/mlp/linear/w_1/Read/ReadVariableOpQenformer/trunk/transformer/transformer_block_3/mlp/linear/b_1/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_3/mlp/linear/w/Read/ReadVariableOpOenformer/trunk/transformer/transformer_block_3/mlp/linear/b/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpyenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpyenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpyenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1/Read/ReadVariableOpyenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1/Read/ReadVariableOpzenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter/Read/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden/Read/ReadVariableOpxenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/b/Read/ReadVariableOp^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/b/Read/ReadVariableOp^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/b/Read/ReadVariableOp^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w/Read/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/w/Read/ReadVariableOpdenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/b/Read/ReadVariableOp^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w/Read/ReadVariableOpConst*?
Tin?
?2?																				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_4769026
?l
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"enformer/heads/head_yeast/output/w"enformer/heads/head_yeast/output/benformer/trunk/stem/conv1_d/wenformer/trunk/stem/conv1_d/bHenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scaleIenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset3enformer/trunk/final_pointwise/conv_block/conv1_d/w3enformer/trunk/final_pointwise/conv_block/conv1_d/bGenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scaleHenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset2enformer/trunk/stem/pointwise_conv_block/conv1_d/w2enformer/trunk/stem/pointwise_conv_block/conv1_d/bNenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter_1Menformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden_1Nenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average_1Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counterKenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hiddenLenformer/trunk/final_pointwise/conv_block/exponential_moving_average/averageMenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter_1Lenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden_1Menformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average_1Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counterJenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hiddenKenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/averageVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/bVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/bVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/bVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scaleWenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offsetAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/wAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter_1[enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden_1\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counterYenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hiddenZenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter_1[enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden_1\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counterYenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hiddenZenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter_1[enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden_1\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counterYenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hiddenZenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter_1[enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden_1\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average_1Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counterYenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hiddenZenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scaleaenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offsetKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/wKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/bCenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_0/mha/layer_norm/offsetGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_biasGenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_biasCenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_0/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_0/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_0/mlp/linear/w;enformer/trunk/transformer/transformer_block_0/mlp/linear/bCenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_1/mha/layer_norm/offsetGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_biasGenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_biasCenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_1/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_1/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_1/mlp/linear/w;enformer/trunk/transformer/transformer_block_1/mlp/linear/bCenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_2/mha/layer_norm/offsetGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_biasGenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_biasCenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_2/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_2/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_2/mlp/linear/w;enformer/trunk/transformer/transformer_block_2/mlp/linear/bCenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scaleDenformer/trunk/transformer/transformer_block_3/mha/layer_norm/offsetGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_biasGenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_biasCenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scaleDenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset=enformer/trunk/transformer/transformer_block_3/mlp/linear/w_1=enformer/trunk/transformer/transformer_block_3/mlp/linear/b_1;enformer/trunk/transformer/transformer_block_3/mlp/linear/w;enformer/trunk/transformer/transformer_block_3/mlp/linear/bfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter_1eenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden_1fenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/countercenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hiddendenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/averagefenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter_1eenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden_1fenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/countercenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hiddendenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/averagefenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter_1eenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden_1fenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/countercenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hiddendenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/averagefenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter_1eenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden_1fenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average_1denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/countercenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hiddendenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/averageHenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/wHenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/wHenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/wPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/wPenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/bJenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/wHenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/wHenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/wHenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/wPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/wPenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/bJenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/wHenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/wHenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/wHenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/wPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/wPenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/bJenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/wHenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/wHenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/wHenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/wPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/wPenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/bJenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_4769540??/
?
h
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768459

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling1d_4_layer_call_fn_4768451

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768404v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768446

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
??
 __inference__traced_save_4769026
file_prefixA
=savev2_enformer_heads_head_yeast_output_w_read_readvariableopA
=savev2_enformer_heads_head_yeast_output_b_read_readvariableop<
8savev2_enformer_trunk_stem_conv1_d_w_read_readvariableop<
8savev2_enformer_trunk_stem_conv1_d_b_read_readvariableopg
csavev2_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableoph
dsavev2_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopR
Nsavev2_enformer_trunk_final_pointwise_conv_block_conv1_d_w_read_readvariableopR
Nsavev2_enformer_trunk_final_pointwise_conv_block_conv1_d_b_read_readvariableopf
bsavev2_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableopg
csavev2_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopQ
Msavev2_enformer_trunk_stem_pointwise_conv_block_conv1_d_w_read_readvariableopQ
Msavev2_enformer_trunk_stem_pointwise_conv_block_conv1_d_b_read_readvariableopm
isavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	l
hsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableopm
isavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopk
gsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	j
fsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopk
gsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_read_readvariableopl
hsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	k
gsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableopl
hsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopj
fsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	i
esavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopj
fsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_read_readvariableopu
qsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_scale_read_readvariableopv
rsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_offset_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_w_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_b_read_readvariableopu
qsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_scale_read_readvariableopv
rsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_offset_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_w_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_b_read_readvariableopu
qsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_scale_read_readvariableopv
rsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_offset_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_w_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_b_read_readvariableopu
qsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_scale_read_readvariableopv
rsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_offset_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_w_read_readvariableop`
\savev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_b_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_1_read_readvariableop	z
vsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_1_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_1_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_read_readvariableop	x
tsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_read_readvariableop
{savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop?
|savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_w_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_b_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_1_read_readvariableop	z
vsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_1_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_1_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_read_readvariableop	x
tsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_read_readvariableop
{savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop?
|savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_w_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_b_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_1_read_readvariableop	z
vsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_1_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_1_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_read_readvariableop	x
tsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_read_readvariableop
{savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop?
|savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_w_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_b_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_1_read_readvariableop	z
vsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_1_read_readvariableop{
wsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_1_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_read_readvariableop	x
tsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_read_readvariableopy
usavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_read_readvariableop
{savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop?
|savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_w_read_readvariableopj
fsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_b_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_offset_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_w_bias_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_r_bias_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_offset_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_1_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_1_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_offset_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_w_bias_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_r_bias_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_offset_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_1_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_1_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_offset_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_w_bias_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_r_bias_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_offset_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_1_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_1_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_offset_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_w_bias_read_readvariableopf
bsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_r_bias_read_readvariableopb
^savev2_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_scale_read_readvariableopc
_savev2_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_offset_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_1_read_readvariableop\
Xsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_1_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_read_readvariableopZ
Vsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	?
?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	?
~savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	?
?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	?
~savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	?
?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	?
~savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop	?
?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?
?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop	?
~savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableop?
savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_b_read_readvariableopi
esavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_b_read_readvariableopi
esavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_b_read_readvariableopi
esavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_w_read_readvariableopg
csavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_w_read_readvariableopo
ksavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_b_read_readvariableopi
esavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_w_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?x
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?w
value?wB?w?B3_heads/yeast/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEB3_heads/yeast/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEB7_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUEB7_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUEBE_trunk/_layers/4/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBF_trunk/_layers/4/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBA_trunk/_layers/4/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBA_trunk/_layers/4/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBM_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBN_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBI_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBI_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBT_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/4/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEB`_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEB__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEB__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/0/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/0/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/1/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/1/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/2/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/2/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/3/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/3/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ??
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_enformer_heads_head_yeast_output_w_read_readvariableop=savev2_enformer_heads_head_yeast_output_b_read_readvariableop8savev2_enformer_trunk_stem_conv1_d_w_read_readvariableop8savev2_enformer_trunk_stem_conv1_d_b_read_readvariableopcsavev2_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableopdsavev2_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopNsavev2_enformer_trunk_final_pointwise_conv_block_conv1_d_w_read_readvariableopNsavev2_enformer_trunk_final_pointwise_conv_block_conv1_d_b_read_readvariableopbsavev2_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableopcsavev2_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopMsavev2_enformer_trunk_stem_pointwise_conv_block_conv1_d_w_read_readvariableopMsavev2_enformer_trunk_stem_pointwise_conv_block_conv1_d_b_read_readvariableopisavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableophsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableopisavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopgsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_read_readvariableopfsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopgsavev2_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_read_readvariableophsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableopgsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableophsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopfsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_read_readvariableopesavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopfsavev2_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_read_readvariableopqsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_scale_read_readvariableoprsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_offset_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_w_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_b_read_readvariableopqsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_scale_read_readvariableoprsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_offset_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_w_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_b_read_readvariableopqsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_scale_read_readvariableoprsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_offset_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_w_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_b_read_readvariableopqsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_scale_read_readvariableoprsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_offset_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_w_read_readvariableop\savev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_b_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_1_read_readvariableopvsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_1_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_1_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_read_readvariableoptsavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_read_readvariableop{savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop|savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_w_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_b_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_1_read_readvariableopvsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_1_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_1_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_read_readvariableoptsavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_read_readvariableop{savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop|savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_w_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_b_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_1_read_readvariableopvsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_1_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_1_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_read_readvariableoptsavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_read_readvariableop{savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop|savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_w_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_b_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_1_read_readvariableopvsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_1_read_readvariableopwsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_1_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_read_readvariableoptsavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_read_readvariableopusavev2_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_read_readvariableop{savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_scale_read_readvariableop|savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_offset_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_w_read_readvariableopfsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_b_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_offset_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_w_bias_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_r_bias_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_offset_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_1_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_1_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_offset_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_w_bias_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_r_bias_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_offset_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_1_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_1_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_offset_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_w_bias_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_r_bias_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_offset_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_1_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_1_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_offset_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_w_bias_read_readvariableopbsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_r_bias_read_readvariableop^savev2_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_scale_read_readvariableop_savev2_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_offset_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_1_read_readvariableopXsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_1_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_read_readvariableopVsavev2_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop~savev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop~savev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop~savev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_1_read_readvariableop?savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_1_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_read_readvariableop~savev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_read_readvariableopsavev2_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_b_read_readvariableopesavev2_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_b_read_readvariableopesavev2_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_b_read_readvariableopesavev2_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_w_read_readvariableopcsavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_w_read_readvariableopksavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_b_read_readvariableopesavev2_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_w_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?																				?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::0:0:?:?:??:?:0:0:00:0: :?:?: :?:?: :0:0: :0:0:0:0:0@:@:@:@:@@:@:@:@:@?:?:?:?:??:?: :0:0: :0:0:@:@:@@:@: :@:@: :@:@:@:@:@@:@: :@:@: :@:@:?:?:??:?: :?:?: :?:?:?:?:??:?:?:?:@:@:?:?:
??:?:
??:?:?:?:@:@:?:?:
??:?:
??:?:?:?:@:@:?:?:
??:?:
??:?:?:?:@:@:?:?:
??:?:
??:?: :@:@: :@:@: :@:@: :@:@: :?:?: :?:?: :?:?: :?:?:
??:
??:
??:
??:?:	?:
??:
??:
??:
??:?:	?:
??:
??:
??:
??:?:	?:
??:
??:
??:
??:?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::($
"
_output_shapes
:0: 

_output_shapes
:0:!

_output_shapes	
:?:!

_output_shapes	
:?:*&
$
_output_shapes
:??:!

_output_shapes	
:?: 	

_output_shapes
:0: 


_output_shapes
:0:($
"
_output_shapes
:00: 

_output_shapes
:0:

_output_shapes
: :)%
#
_output_shapes
:?:)%
#
_output_shapes
:?:

_output_shapes
: :)%
#
_output_shapes
:?:)%
#
_output_shapes
:?:

_output_shapes
: :($
"
_output_shapes
:0:($
"
_output_shapes
:0:

_output_shapes
: :($
"
_output_shapes
:0:($
"
_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:($
"
_output_shapes
:0@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@:)#%
#
_output_shapes
:@?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:*'&
$
_output_shapes
:??:!(

_output_shapes	
:?:)

_output_shapes
: :(*$
"
_output_shapes
:0:(+$
"
_output_shapes
:0:,

_output_shapes
: :(-$
"
_output_shapes
:0:(.$
"
_output_shapes
:0: /

_output_shapes
:@: 0

_output_shapes
:@:(1$
"
_output_shapes
:@@: 2

_output_shapes
:@:3

_output_shapes
: :(4$
"
_output_shapes
:@:(5$
"
_output_shapes
:@:6

_output_shapes
: :(7$
"
_output_shapes
:@:(8$
"
_output_shapes
:@: 9

_output_shapes
:@: :

_output_shapes
:@:(;$
"
_output_shapes
:@@: <

_output_shapes
:@:=

_output_shapes
: :(>$
"
_output_shapes
:@:(?$
"
_output_shapes
:@:@

_output_shapes
: :(A$
"
_output_shapes
:@:(B$
"
_output_shapes
:@:!C

_output_shapes	
:?:!D

_output_shapes	
:?:*E&
$
_output_shapes
:??:!F

_output_shapes	
:?:G

_output_shapes
: :)H%
#
_output_shapes
:?:)I%
#
_output_shapes
:?:J

_output_shapes
: :)K%
#
_output_shapes
:?:)L%
#
_output_shapes
:?:!M

_output_shapes	
:?:!N

_output_shapes	
:?:*O&
$
_output_shapes
:??:!P

_output_shapes	
:?:!Q

_output_shapes	
:?:!R

_output_shapes	
:?:,S(
&
_output_shapes
:@:,T(
&
_output_shapes
:@:!U

_output_shapes	
:?:!V

_output_shapes	
:?:&W"
 
_output_shapes
:
??:!X

_output_shapes	
:?:&Y"
 
_output_shapes
:
??:!Z

_output_shapes	
:?:![

_output_shapes	
:?:!\

_output_shapes	
:?:,](
&
_output_shapes
:@:,^(
&
_output_shapes
:@:!_

_output_shapes	
:?:!`

_output_shapes	
:?:&a"
 
_output_shapes
:
??:!b

_output_shapes	
:?:&c"
 
_output_shapes
:
??:!d

_output_shapes	
:?:!e

_output_shapes	
:?:!f

_output_shapes	
:?:,g(
&
_output_shapes
:@:,h(
&
_output_shapes
:@:!i

_output_shapes	
:?:!j

_output_shapes	
:?:&k"
 
_output_shapes
:
??:!l

_output_shapes	
:?:&m"
 
_output_shapes
:
??:!n

_output_shapes	
:?:!o

_output_shapes	
:?:!p

_output_shapes	
:?:,q(
&
_output_shapes
:@:,r(
&
_output_shapes
:@:!s

_output_shapes	
:?:!t

_output_shapes	
:?:&u"
 
_output_shapes
:
??:!v

_output_shapes	
:?:&w"
 
_output_shapes
:
??:!x

_output_shapes	
:?:y

_output_shapes
: :(z$
"
_output_shapes
:@:({$
"
_output_shapes
:@:|

_output_shapes
: :(}$
"
_output_shapes
:@:(~$
"
_output_shapes
:@:

_output_shapes
: :)?$
"
_output_shapes
:@:)?$
"
_output_shapes
:@:?

_output_shapes
: :)?$
"
_output_shapes
:@:)?$
"
_output_shapes
:@:?

_output_shapes
: :*?%
#
_output_shapes
:?:*?%
#
_output_shapes
:?:?

_output_shapes
: :*?%
#
_output_shapes
:?:*?%
#
_output_shapes
:?:?

_output_shapes
: :*?%
#
_output_shapes
:?:*?%
#
_output_shapes
:?:?

_output_shapes
: :*?%
#
_output_shapes
:?:*?%
#
_output_shapes
:?:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:'?"
 
_output_shapes
:
??:"?

_output_shapes	
:?:&?!

_output_shapes
:	?:?

_output_shapes
: 
?5
?
%__inference_signature_wrapper_4768291

args_0
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
	unknown_3:0
	unknown_4:0
	unknown_5:00
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0 

unknown_11:0@

unknown_12:@ 

unknown_13:@ 

unknown_14:@

unknown_15:@

unknown_16:@ 

unknown_17:@@

unknown_18:@ 

unknown_19:@ 

unknown_20:@

unknown_21:@

unknown_22:@ 

unknown_23:@@

unknown_24:@ 

unknown_25:@ 

unknown_26:@

unknown_27:@

unknown_28:@ 

unknown_29:@@

unknown_30:@ 

unknown_31:@ 

unknown_32:@

unknown_33:@

unknown_34:@!

unknown_35:@?

unknown_36:	?!

unknown_37:?!

unknown_38:?

unknown_39:	?

unknown_40:	?"

unknown_41:??

unknown_42:	?!

unknown_43:?!

unknown_44:?

unknown_45:	?

unknown_46:	?"

unknown_47:??

unknown_48:	?!

unknown_49:?!

unknown_50:?

unknown_51:	?

unknown_52:	?"

unknown_53:??

unknown_54:	?

unknown_55:	?

unknown_56:	?

unknown_57:
??

unknown_58:
??

unknown_59:
??

unknown_60:	?$

unknown_61:@$

unknown_62:@

unknown_63:
??

unknown_64:	?

unknown_65:	?

unknown_66:	?

unknown_67:
??

unknown_68:	?

unknown_69:
??

unknown_70:	?

unknown_71:	?

unknown_72:	?

unknown_73:
??

unknown_74:
??

unknown_75:
??

unknown_76:	?$

unknown_77:@$

unknown_78:@

unknown_79:
??

unknown_80:	?

unknown_81:	?

unknown_82:	?

unknown_83:
??

unknown_84:	?

unknown_85:
??

unknown_86:	?

unknown_87:	?

unknown_88:	?

unknown_89:
??

unknown_90:
??

unknown_91:
??

unknown_92:	?$

unknown_93:@$

unknown_94:@

unknown_95:
??

unknown_96:	?

unknown_97:	?

unknown_98:	?

unknown_99:
??
unknown_100:	?
unknown_101:
??
unknown_102:	?
unknown_103:	?
unknown_104:	?
unknown_105:
??
unknown_106:
??
unknown_107:
??
unknown_108:	?%
unknown_109:@%
unknown_110:@
unknown_111:
??
unknown_112:	?
unknown_113:	?
unknown_114:	?
unknown_115:
??
unknown_116:	?
unknown_117:
??
unknown_118:	?"
unknown_119:?"
unknown_120:?
unknown_121:	?
unknown_122:	?#
unknown_123:??
unknown_124:	?
unknown_125:	?
unknown_126:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102unknown_103unknown_104unknown_105unknown_106unknown_107unknown_108unknown_109unknown_110unknown_111unknown_112unknown_113unknown_114unknown_115unknown_116unknown_117unknown_118unknown_119unknown_120unknown_121unknown_122unknown_123unknown_124unknown_125unknown_126*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*?
_read_only_resource_inputs?
??	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_predict_on_batch_791105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameargs_0
?
h
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768318

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768303

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling1d_1_layer_call_fn_4768412

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768359v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
ȹ
#__inference__traced_restore_4769540
file_prefixF
3assignvariableop_enformer_heads_head_yeast_output_w:	?C
5assignvariableop_1_enformer_heads_head_yeast_output_b:F
0assignvariableop_2_enformer_trunk_stem_conv1_d_w:0>
0assignvariableop_3_enformer_trunk_stem_conv1_d_b:0j
[assignvariableop_4_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_scale:	?k
\assignvariableop_5_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_offset:	?^
Fassignvariableop_6_enformer_trunk_final_pointwise_conv_block_conv1_d_w:??U
Fassignvariableop_7_enformer_trunk_final_pointwise_conv_block_conv1_d_b:	?h
Zassignvariableop_8_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_scale:0i
[assignvariableop_9_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_offset:0\
Fassignvariableop_10_enformer_trunk_stem_pointwise_conv_block_conv1_d_w:00T
Fassignvariableop_11_enformer_trunk_stem_pointwise_conv_block_conv1_d_b:0l
bassignvariableop_12_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_1:	 x
aassignvariableop_13_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_1:?y
bassignvariableop_14_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_1:?j
`assignvariableop_15_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter:	 v
_assignvariableop_16_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden:?w
`assignvariableop_17_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average:?k
aassignvariableop_18_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_1:	 v
`assignvariableop_19_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_1:0w
aassignvariableop_20_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_1:0i
_assignvariableop_21_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter:	 t
^assignvariableop_22_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden:0u
_assignvariableop_23_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average:0x
jassignvariableop_24_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_scale:0y
kassignvariableop_25_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_offset:0k
Uassignvariableop_26_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_w:0@c
Uassignvariableop_27_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_b:@x
jassignvariableop_28_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_scale:@y
kassignvariableop_29_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_offset:@k
Uassignvariableop_30_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_w:@@c
Uassignvariableop_31_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_b:@x
jassignvariableop_32_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_scale:@y
kassignvariableop_33_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_offset:@l
Uassignvariableop_34_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_w:@?d
Uassignvariableop_35_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_b:	?y
jassignvariableop_36_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_scale:	?z
kassignvariableop_37_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_offset:	?m
Uassignvariableop_38_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_w:??d
Uassignvariableop_39_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_b:	?z
passignvariableop_40_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_1:	 ?
oassignvariableop_41_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_1:0?
passignvariableop_42_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_1:0x
nassignvariableop_43_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter:	 ?
massignvariableop_44_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden:0?
nassignvariableop_45_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average:0?
tassignvariableop_46_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_scale:@?
uassignvariableop_47_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_offset:@u
_assignvariableop_48_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_w:@@m
_assignvariableop_49_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_b:@z
passignvariableop_50_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_1:	 ?
oassignvariableop_51_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_1:@?
passignvariableop_52_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_1:@x
nassignvariableop_53_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter:	 ?
massignvariableop_54_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden:@?
nassignvariableop_55_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average:@?
tassignvariableop_56_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_scale:@?
uassignvariableop_57_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_offset:@u
_assignvariableop_58_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_w:@@m
_assignvariableop_59_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_b:@z
passignvariableop_60_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_1:	 ?
oassignvariableop_61_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_1:@?
passignvariableop_62_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_1:@x
nassignvariableop_63_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter:	 ?
massignvariableop_64_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden:@?
nassignvariableop_65_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average:@?
tassignvariableop_66_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_scale:	??
uassignvariableop_67_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_offset:	?w
_assignvariableop_68_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_w:??n
_assignvariableop_69_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_b:	?z
passignvariableop_70_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_1:	 ?
oassignvariableop_71_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_1:??
passignvariableop_72_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_1:?x
nassignvariableop_73_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter:	 ?
massignvariableop_74_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden:??
nassignvariableop_75_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average:??
tassignvariableop_76_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_scale:	??
uassignvariableop_77_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_offset:	?w
_assignvariableop_78_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_w:??n
_assignvariableop_79_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_b:	?f
Wassignvariableop_80_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_scale:	?g
Xassignvariableop_81_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_offset:	?u
[assignvariableop_82_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_w_bias:@u
[assignvariableop_83_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_r_bias:@f
Wassignvariableop_84_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_scale:	?g
Xassignvariableop_85_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_offset:	?e
Qassignvariableop_86_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_1:
??`
Qassignvariableop_87_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_1:	?c
Oassignvariableop_88_enformer_trunk_transformer_transformer_block_0_mlp_linear_w:
??^
Oassignvariableop_89_enformer_trunk_transformer_transformer_block_0_mlp_linear_b:	?f
Wassignvariableop_90_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_scale:	?g
Xassignvariableop_91_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_offset:	?u
[assignvariableop_92_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_w_bias:@u
[assignvariableop_93_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_r_bias:@f
Wassignvariableop_94_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_scale:	?g
Xassignvariableop_95_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_offset:	?e
Qassignvariableop_96_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_1:
??`
Qassignvariableop_97_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_1:	?c
Oassignvariableop_98_enformer_trunk_transformer_transformer_block_1_mlp_linear_w:
??^
Oassignvariableop_99_enformer_trunk_transformer_transformer_block_1_mlp_linear_b:	?g
Xassignvariableop_100_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_scale:	?h
Yassignvariableop_101_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_offset:	?v
\assignvariableop_102_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_w_bias:@v
\assignvariableop_103_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_r_bias:@g
Xassignvariableop_104_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_scale:	?h
Yassignvariableop_105_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_offset:	?f
Rassignvariableop_106_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_1:
??a
Rassignvariableop_107_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_1:	?d
Passignvariableop_108_enformer_trunk_transformer_transformer_block_2_mlp_linear_w:
??_
Passignvariableop_109_enformer_trunk_transformer_transformer_block_2_mlp_linear_b:	?g
Xassignvariableop_110_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_scale:	?h
Yassignvariableop_111_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_offset:	?v
\assignvariableop_112_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_w_bias:@v
\assignvariableop_113_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_r_bias:@g
Xassignvariableop_114_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_scale:	?h
Yassignvariableop_115_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_offset:	?f
Rassignvariableop_116_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_1:
??a
Rassignvariableop_117_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_1:	?d
Passignvariableop_118_enformer_trunk_transformer_transformer_block_3_mlp_linear_w:
??_
Passignvariableop_119_enformer_trunk_transformer_transformer_block_3_mlp_linear_b:	??
{assignvariableop_120_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_1:	 ?
zassignvariableop_121_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_1:@?
{assignvariableop_122_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_1:@?
yassignvariableop_123_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter:	 ?
xassignvariableop_124_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden:@?
yassignvariableop_125_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average:@?
{assignvariableop_126_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_1:	 ?
zassignvariableop_127_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_1:@?
{assignvariableop_128_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_1:@?
yassignvariableop_129_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter:	 ?
xassignvariableop_130_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden:@?
yassignvariableop_131_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average:@?
{assignvariableop_132_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_1:	 ?
zassignvariableop_133_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_1:??
{assignvariableop_134_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_1:??
yassignvariableop_135_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter:	 ?
xassignvariableop_136_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden:??
yassignvariableop_137_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average:??
{assignvariableop_138_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_1:	 ?
zassignvariableop_139_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_1:??
{assignvariableop_140_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_1:??
yassignvariableop_141_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter:	 ?
xassignvariableop_142_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden:??
yassignvariableop_143_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average:?q
]assignvariableop_144_enformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_w:
??q
]assignvariableop_145_enformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_w:
??q
]assignvariableop_146_enformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_w:
??y
eassignvariableop_147_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_w:
??t
eassignvariableop_148_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_b:	?r
_assignvariableop_149_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_w:	?q
]assignvariableop_150_enformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_w:
??q
]assignvariableop_151_enformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_w:
??q
]assignvariableop_152_enformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_w:
??y
eassignvariableop_153_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_w:
??t
eassignvariableop_154_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_b:	?r
_assignvariableop_155_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_w:	?q
]assignvariableop_156_enformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_w:
??q
]assignvariableop_157_enformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_w:
??q
]assignvariableop_158_enformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_w:
??y
eassignvariableop_159_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_w:
??t
eassignvariableop_160_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_b:	?r
_assignvariableop_161_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_w:	?q
]assignvariableop_162_enformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_w:
??q
]assignvariableop_163_enformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_w:
??q
]assignvariableop_164_enformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_w:
??y
eassignvariableop_165_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_w:
??t
eassignvariableop_166_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_b:	?r
_assignvariableop_167_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_w:	?
identity_169??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_159?AssignVariableOp_16?AssignVariableOp_160?AssignVariableOp_161?AssignVariableOp_162?AssignVariableOp_163?AssignVariableOp_164?AssignVariableOp_165?AssignVariableOp_166?AssignVariableOp_167?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?x
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?w
value?wB?w?B3_heads/yeast/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEB3_heads/yeast/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEB7_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUEB7_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUEBE_trunk/_layers/4/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBF_trunk/_layers/4/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBA_trunk/_layers/4/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBA_trunk/_layers/4/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBM_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBN_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBI_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBI_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBT_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/4/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/4/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/4/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEB`_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEB__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEB__trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/0/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/0/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/0/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/1/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/1/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/1/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/2/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/2/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/2/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBO_trunk/_layers/1/_layers/3/_layers/0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBP_trunk/_layers/1/_layers/3/_layers/0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBK_trunk/_layers/1/_layers/3/_layers/0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/0/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/1/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/2/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEB]_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBb_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBa_trunk/_layers/1/_layers/3/_layers/0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/0/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/1/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/2/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_w_bias/.ATTRIBUTES/VARIABLE_VALUEB[_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_r_bias/.ATTRIBUTES/VARIABLE_VALUEBW_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUEBX_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/1/b/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/w/.ATTRIBUTES/VARIABLE_VALUEBS_trunk/_layers/2/_layers/3/_layers/1/_module/_layers/4/b/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/1/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/2/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEBf_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUEBe_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUEBj_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUEBi_trunk/_layers/1/_layers/3/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/0/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/1/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/2/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_q_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB\_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_v_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUEBd_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUEB^_trunk/_layers/2/_layers/3/_layers/0/_module/_layers/1/_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?																				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp3assignvariableop_enformer_heads_head_yeast_output_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp5assignvariableop_1_enformer_heads_head_yeast_output_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp0assignvariableop_2_enformer_trunk_stem_conv1_d_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp0assignvariableop_3_enformer_trunk_stem_conv1_d_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp[assignvariableop_4_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp\assignvariableop_5_enformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpFassignvariableop_6_enformer_trunk_final_pointwise_conv_block_conv1_d_wIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpFassignvariableop_7_enformer_trunk_final_pointwise_conv_block_conv1_d_bIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpZassignvariableop_8_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp[assignvariableop_9_enformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpFassignvariableop_10_enformer_trunk_stem_pointwise_conv_block_conv1_d_wIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpFassignvariableop_11_enformer_trunk_stem_pointwise_conv_block_conv1_d_bIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpbassignvariableop_12_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counter_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpaassignvariableop_13_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hidden_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpbassignvariableop_14_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_average_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp`assignvariableop_15_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_counterIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp_assignvariableop_16_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_hiddenIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp`assignvariableop_17_enformer_trunk_final_pointwise_conv_block_exponential_moving_average_averageIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpaassignvariableop_18_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counter_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp`assignvariableop_19_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hidden_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpaassignvariableop_20_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_average_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp_assignvariableop_21_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_counterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp^assignvariableop_22_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_hiddenIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp_assignvariableop_23_enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_averageIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpjassignvariableop_24_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_scaleIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpkassignvariableop_25_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_offsetIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpUassignvariableop_26_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_wIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpUassignvariableop_27_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_bIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpjassignvariableop_28_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_scaleIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpkassignvariableop_29_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_offsetIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpUassignvariableop_30_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_wIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpUassignvariableop_31_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_bIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpjassignvariableop_32_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_scaleIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpkassignvariableop_33_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_offsetIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpUassignvariableop_34_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_wIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpUassignvariableop_35_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_bIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpjassignvariableop_36_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_scaleIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpkassignvariableop_37_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_offsetIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpUassignvariableop_38_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_wIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpUassignvariableop_39_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_bIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_40AssignVariableOppassignvariableop_40_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counter_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpoassignvariableop_41_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hidden_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOppassignvariableop_42_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_average_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpnassignvariableop_43_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_counterIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpmassignvariableop_44_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_hiddenIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpnassignvariableop_45_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_averageIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOptassignvariableop_46_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpuassignvariableop_47_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp_assignvariableop_48_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_wIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp_assignvariableop_49_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_bIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_50AssignVariableOppassignvariableop_50_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counter_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpoassignvariableop_51_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hidden_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOppassignvariableop_52_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_average_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpnassignvariableop_53_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_counterIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpmassignvariableop_54_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_hiddenIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpnassignvariableop_55_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_averageIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOptassignvariableop_56_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpuassignvariableop_57_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp_assignvariableop_58_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_wIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp_assignvariableop_59_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_bIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_60AssignVariableOppassignvariableop_60_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counter_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpoassignvariableop_61_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hidden_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOppassignvariableop_62_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_average_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpnassignvariableop_63_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_counterIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpmassignvariableop_64_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_hiddenIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpnassignvariableop_65_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_averageIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOptassignvariableop_66_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpuassignvariableop_67_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp_assignvariableop_68_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_wIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp_assignvariableop_69_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_bIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_70AssignVariableOppassignvariableop_70_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counter_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpoassignvariableop_71_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hidden_1Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOppassignvariableop_72_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_average_1Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpnassignvariableop_73_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_counterIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpmassignvariableop_74_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_hiddenIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOpnassignvariableop_75_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_averageIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOptassignvariableop_76_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_scaleIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpuassignvariableop_77_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_offsetIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp_assignvariableop_78_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_wIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp_assignvariableop_79_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_bIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOpWassignvariableop_80_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_scaleIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOpXassignvariableop_81_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_offsetIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp[assignvariableop_82_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_w_biasIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp[assignvariableop_83_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_r_biasIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOpWassignvariableop_84_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_scaleIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOpXassignvariableop_85_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_offsetIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOpQassignvariableop_86_enformer_trunk_transformer_transformer_block_0_mlp_linear_w_1Identity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOpQassignvariableop_87_enformer_trunk_transformer_transformer_block_0_mlp_linear_b_1Identity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOpOassignvariableop_88_enformer_trunk_transformer_transformer_block_0_mlp_linear_wIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOpOassignvariableop_89_enformer_trunk_transformer_transformer_block_0_mlp_linear_bIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOpWassignvariableop_90_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_scaleIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOpXassignvariableop_91_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_offsetIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp[assignvariableop_92_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_w_biasIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp[assignvariableop_93_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_r_biasIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOpWassignvariableop_94_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_scaleIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOpXassignvariableop_95_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_offsetIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOpQassignvariableop_96_enformer_trunk_transformer_transformer_block_1_mlp_linear_w_1Identity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOpQassignvariableop_97_enformer_trunk_transformer_transformer_block_1_mlp_linear_b_1Identity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOpOassignvariableop_98_enformer_trunk_transformer_transformer_block_1_mlp_linear_wIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOpOassignvariableop_99_enformer_trunk_transformer_transformer_block_1_mlp_linear_bIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOpXassignvariableop_100_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_scaleIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOpYassignvariableop_101_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_offsetIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp\assignvariableop_102_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_w_biasIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp\assignvariableop_103_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_r_biasIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOpXassignvariableop_104_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_scaleIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOpYassignvariableop_105_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_offsetIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOpRassignvariableop_106_enformer_trunk_transformer_transformer_block_2_mlp_linear_w_1Identity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOpRassignvariableop_107_enformer_trunk_transformer_transformer_block_2_mlp_linear_b_1Identity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOpPassignvariableop_108_enformer_trunk_transformer_transformer_block_2_mlp_linear_wIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOpPassignvariableop_109_enformer_trunk_transformer_transformer_block_2_mlp_linear_bIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOpXassignvariableop_110_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_scaleIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOpYassignvariableop_111_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_offsetIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp\assignvariableop_112_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_w_biasIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp\assignvariableop_113_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_r_biasIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOpXassignvariableop_114_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_scaleIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOpYassignvariableop_115_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_offsetIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOpRassignvariableop_116_enformer_trunk_transformer_transformer_block_3_mlp_linear_w_1Identity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOpRassignvariableop_117_enformer_trunk_transformer_transformer_block_3_mlp_linear_b_1Identity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOpPassignvariableop_118_enformer_trunk_transformer_transformer_block_3_mlp_linear_wIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOpPassignvariableop_119_enformer_trunk_transformer_transformer_block_3_mlp_linear_bIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp{assignvariableop_120_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counter_1Identity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOpzassignvariableop_121_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hidden_1Identity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp{assignvariableop_122_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_average_1Identity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_123AssignVariableOpyassignvariableop_123_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_counterIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOpxassignvariableop_124_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_hiddenIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOpyassignvariableop_125_enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_averageIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp{assignvariableop_126_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counter_1Identity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOpzassignvariableop_127_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hidden_1Identity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp{assignvariableop_128_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_average_1Identity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_129AssignVariableOpyassignvariableop_129_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_counterIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_130AssignVariableOpxassignvariableop_130_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_hiddenIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_131AssignVariableOpyassignvariableop_131_enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_averageIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_132AssignVariableOp{assignvariableop_132_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counter_1Identity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_133AssignVariableOpzassignvariableop_133_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hidden_1Identity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_134AssignVariableOp{assignvariableop_134_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_average_1Identity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_135AssignVariableOpyassignvariableop_135_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_counterIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_136AssignVariableOpxassignvariableop_136_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_hiddenIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_137AssignVariableOpyassignvariableop_137_enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_averageIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_138AssignVariableOp{assignvariableop_138_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counter_1Identity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_139AssignVariableOpzassignvariableop_139_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hidden_1Identity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_140AssignVariableOp{assignvariableop_140_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_average_1Identity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_141AssignVariableOpyassignvariableop_141_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_counterIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_142AssignVariableOpxassignvariableop_142_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_hiddenIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_143AssignVariableOpyassignvariableop_143_enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_averageIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_144AssignVariableOp]assignvariableop_144_enformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_wIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_145AssignVariableOp]assignvariableop_145_enformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_wIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_146AssignVariableOp]assignvariableop_146_enformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_wIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_147AssignVariableOpeassignvariableop_147_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_wIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_148AssignVariableOpeassignvariableop_148_enformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_bIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_149AssignVariableOp_assignvariableop_149_enformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_wIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_150AssignVariableOp]assignvariableop_150_enformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_wIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_151AssignVariableOp]assignvariableop_151_enformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_wIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_152AssignVariableOp]assignvariableop_152_enformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_wIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_153AssignVariableOpeassignvariableop_153_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_wIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_154AssignVariableOpeassignvariableop_154_enformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_bIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_155AssignVariableOp_assignvariableop_155_enformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_wIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_156AssignVariableOp]assignvariableop_156_enformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_wIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_157AssignVariableOp]assignvariableop_157_enformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_wIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_158AssignVariableOp]assignvariableop_158_enformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_wIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_159AssignVariableOpeassignvariableop_159_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_wIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_160AssignVariableOpeassignvariableop_160_enformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_bIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_161AssignVariableOp_assignvariableop_161_enformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_wIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_162AssignVariableOp]assignvariableop_162_enformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_wIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_163AssignVariableOp]assignvariableop_163_enformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_wIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_164AssignVariableOp]assignvariableop_164_enformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_wIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_165AssignVariableOpeassignvariableop_165_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_wIdentity_165:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_166AssignVariableOpeassignvariableop_166_enformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_bIdentity_166:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_167AssignVariableOp_assignvariableop_167_enformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_wIdentity_167:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_168Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_169IdentityIdentity_168:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_169Identity_169:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672*
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
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
h
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768359

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768433

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768334

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768389

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??#
??
#__inference_predict_on_batch_791105

args_0b
Lenformer_trunk_stem_conv1_d_convolution_expanddims_1_readvariableop_resource:0I
;enformer_trunk_stem_conv1_d_biasadd_readvariableop_resource:0v
`enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:0x
benformer_trunk_stem_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:0u
genformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:0q
cenformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:0w
aenformer_trunk_stem_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:00^
Penformer_trunk_stem_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:0?
oenformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_read_readvariableop_resource:0?
qenformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_read_1_readvariableop_resource:0?
venformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:0?
renformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:0?
penformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:0@m
_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_biasadd_readvariableop_resource:@?
yenformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:@?
{enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:@?
?enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:@?
|enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:@?
zenformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:@@w
ienformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:@?
oenformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_read_readvariableop_resource:@?
qenformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_read_1_readvariableop_resource:@?
venformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:@?
renformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:@?
penformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:@@m
_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_biasadd_readvariableop_resource:@?
yenformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:@?
{enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:@?
?enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:@?
|enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:@?
zenformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:@@w
ienformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:@?
oenformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_read_readvariableop_resource:@?
qenformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_read_1_readvariableop_resource:@?
venformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:@?
renformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:@?
penformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:@?n
_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_biasadd_readvariableop_resource:	??
yenformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:??
{enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:??
?enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:	??
|enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:	??
zenformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:??x
ienformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:	??
oenformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_read_readvariableop_resource:??
qenformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_read_1_readvariableop_resource:??
venformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:	??
renformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:	??
penformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:??n
_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_biasadd_readvariableop_resource:	??
yenformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:??
{enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:??
?enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:	??
|enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:	??
zenformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:??x
ienformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_0_mha_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_batchnorm_readvariableop_resource:	?y
eenformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_matmul_readvariableop_resource:
??z
genformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_matmul_readvariableop_resource:	?v
\enformer_trunk_transformer_transformer_block_0_mha_attention_0_add_1_readvariableop_resource:@v
\enformer_trunk_transformer_transformer_block_0_mha_attention_0_add_2_readvariableop_resource:@?
menformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_matmul_readvariableop_resource:
??y
jenformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_add_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_0_mlp_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_batchnorm_readvariableop_resource:	?l
Xenformer_trunk_transformer_transformer_block_0_mlp_linear_matmul_readvariableop_resource:
??d
Uenformer_trunk_transformer_transformer_block_0_mlp_linear_add_readvariableop_resource:	?n
Zenformer_trunk_transformer_transformer_block_0_mlp_linear_matmul_1_readvariableop_resource:
??f
Wenformer_trunk_transformer_transformer_block_0_mlp_linear_add_1_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_1_mha_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_batchnorm_readvariableop_resource:	?y
eenformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_matmul_readvariableop_resource:
??z
genformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_matmul_readvariableop_resource:	?v
\enformer_trunk_transformer_transformer_block_1_mha_attention_1_add_1_readvariableop_resource:@v
\enformer_trunk_transformer_transformer_block_1_mha_attention_1_add_2_readvariableop_resource:@?
menformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_matmul_readvariableop_resource:
??y
jenformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_add_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_1_mlp_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_batchnorm_readvariableop_resource:	?l
Xenformer_trunk_transformer_transformer_block_1_mlp_linear_matmul_readvariableop_resource:
??d
Uenformer_trunk_transformer_transformer_block_1_mlp_linear_add_readvariableop_resource:	?n
Zenformer_trunk_transformer_transformer_block_1_mlp_linear_matmul_1_readvariableop_resource:
??f
Wenformer_trunk_transformer_transformer_block_1_mlp_linear_add_1_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_2_mha_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_batchnorm_readvariableop_resource:	?y
eenformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_matmul_readvariableop_resource:
??z
genformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_matmul_readvariableop_resource:	?v
\enformer_trunk_transformer_transformer_block_2_mha_attention_2_add_1_readvariableop_resource:@v
\enformer_trunk_transformer_transformer_block_2_mha_attention_2_add_2_readvariableop_resource:@?
menformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_matmul_readvariableop_resource:
??y
jenformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_add_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_2_mlp_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_batchnorm_readvariableop_resource:	?l
Xenformer_trunk_transformer_transformer_block_2_mlp_linear_matmul_readvariableop_resource:
??d
Uenformer_trunk_transformer_transformer_block_2_mlp_linear_add_readvariableop_resource:	?n
Zenformer_trunk_transformer_transformer_block_2_mlp_linear_matmul_1_readvariableop_resource:
??f
Wenformer_trunk_transformer_transformer_block_2_mlp_linear_add_1_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_3_mha_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_batchnorm_readvariableop_resource:	?y
eenformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_matmul_readvariableop_resource:
??y
eenformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_matmul_readvariableop_resource:
??z
genformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_matmul_readvariableop_resource:	?v
\enformer_trunk_transformer_transformer_block_3_mha_attention_3_add_1_readvariableop_resource:@v
\enformer_trunk_transformer_transformer_block_3_mha_attention_3_add_2_readvariableop_resource:@?
menformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_matmul_readvariableop_resource:
??y
jenformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_add_readvariableop_resource:	?r
cenformer_trunk_transformer_transformer_block_3_mlp_layer_norm_batchnorm_mul_readvariableop_resource:	?n
_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_batchnorm_readvariableop_resource:	?l
Xenformer_trunk_transformer_transformer_block_3_mlp_linear_matmul_readvariableop_resource:
??d
Uenformer_trunk_transformer_transformer_block_3_mlp_linear_add_readvariableop_resource:	?n
Zenformer_trunk_transformer_transformer_block_3_mlp_linear_matmul_1_readvariableop_resource:
??f
Wenformer_trunk_transformer_transformer_block_3_mlp_linear_add_1_readvariableop_resource:	?x
aenformer_trunk_final_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource:?z
cenformer_trunk_final_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource:?w
henformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource:	?s
denformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource:	?z
benformer_trunk_final_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource:??`
Qenformer_trunk_final_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource:	?R
?enformer_heads_head_yeast_output_matmul_readvariableop_resource:	?J
<enformer_heads_head_yeast_output_add_readvariableop_resource:
identity??3enformer/heads/head_yeast/output/Add/ReadVariableOp?6enformer/heads/head_yeast/output/MatMul/ReadVariableOp?Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOp?genformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?ienformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?menformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?fenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOp?henformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOp?`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp?qenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?senformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?wenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?penformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp?renformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp?Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOp?genformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?ienformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?menformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?fenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOp?henformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOp?`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp?qenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?senformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?wenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?penformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp?renformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp?Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOp?genformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?ienformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?menformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?fenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOp?henformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOp?`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp?qenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?senformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?wenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?penformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp?renformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp?Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOp?genformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?ienformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?menformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?fenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOp?henformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOp?`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp?qenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?senformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?wenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?penformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp?renformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp?Henformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOp?Yenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?[enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?_enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?Xenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOp?Zenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOp?2enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOp?Cenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOp?Genformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp?Xenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp?Zenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp?^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp?Wenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp?Yenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOp?aenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOp?denformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOp?^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOp?Lenformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOp?Nenformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOp?Oenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOp?Qenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOp?aenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOp?denformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOp?^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOp?Lenformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOp?Nenformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOp?Oenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOp?Qenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOp?aenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOp?denformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOp?^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOp?Lenformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOp?Nenformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOp?Oenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOp?Qenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOp?Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOp?aenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOp?denformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOp?^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOp?\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOp?Venformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOp?Zenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOp?Lenformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOp?Nenformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOp?Oenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOp?Qenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOp?
6enformer/trunk/stem/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2enformer/trunk/stem/conv1_d/convolution/ExpandDims
ExpandDimsargs_0?enformer/trunk/stem/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:????????????
Cenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpLenformer_trunk_stem_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:0*
dtype0z
8enformer/trunk/stem/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
4enformer/trunk/stem/conv1_d/convolution/ExpandDims_1
ExpandDimsKenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0Aenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0?
'enformer/trunk/stem/conv1_d/convolutionConv2D;enformer/trunk/stem/conv1_d/convolution/ExpandDims:output:0=enformer/trunk/stem/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
?
/enformer/trunk/stem/conv1_d/convolution/SqueezeSqueeze0enformer/trunk/stem/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????0*
squeeze_dims

??????????
2enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOpReadVariableOp;enformer_trunk_stem_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#enformer/trunk/stem/conv1_d/BiasAddBiasAdd8enformer/trunk/stem/conv1_d/convolution/Squeeze:output:0:enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????0?
Wenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOp`enformer_trunk_stem_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:0*
dtype0?
Lenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/IdentityIdentity_enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
Yenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpbenformer_trunk_stem_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:0*
dtype0?
Nenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Identity_1Identityaenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
Qenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Oenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/addAddV2Wenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Identity_1:output:0Zenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:0?
Qenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtSenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:0?
^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOpgenformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:0*
dtype0?
Oenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mulMulUenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0fenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
Qenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1Mul,enformer/trunk/stem/conv1_d/BiasAdd:output:0Senformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????0?
Qenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2MulUenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Identity:output:0Senformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:0?
Zenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOpcenformer_trunk_stem_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype0?
Oenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/subSubbenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0Uenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:0?
Qenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2Uenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0Senformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????0s
.enformer/trunk/stem/pointwise_conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
,enformer/trunk/stem/pointwise_conv_block/mulMul7enformer/trunk/stem/pointwise_conv_block/mul/x:output:0Uenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????0?
0enformer/trunk/stem/pointwise_conv_block/SigmoidSigmoid0enformer/trunk/stem/pointwise_conv_block/mul:z:0*
T0*-
_output_shapes
:???????????0?
.enformer/trunk/stem/pointwise_conv_block/mul_1Mul4enformer/trunk/stem/pointwise_conv_block/Sigmoid:y:0Uenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????0?
Kenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims
ExpandDims2enformer/trunk/stem/pointwise_conv_block/mul_1:z:0Tenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????0?
Xenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpaenformer_trunk_stem_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:00*
dtype0?
Menformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Ienformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1
ExpandDims`enformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0Venformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:00?
<enformer/trunk/stem/pointwise_conv_block/conv1_d/convolutionConv2DPenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims:output:0Renformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????0*
paddingSAME*
strides
?
Denformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/SqueezeSqueezeEenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????0*
squeeze_dims

??????????
Genformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpPenformer_trunk_stem_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
8enformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAddBiasAddMenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/Squeeze:output:0Oenformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????0?
 enformer/trunk/stem/residual/addAddV2,enformer/trunk/stem/conv1_d/BiasAdd:output:0Aenformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd:output:0*
T0*-
_output_shapes
:???????????0r
0enformer/trunk/stem/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
,enformer/trunk/stem/max_pooling1d/ExpandDims
ExpandDims$enformer/trunk/stem/residual/add:z:09enformer/trunk/stem/max_pooling1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????0?
)enformer/trunk/stem/max_pooling1d/MaxPoolMaxPool5enformer/trunk/stem/max_pooling1d/ExpandDims:output:0*1
_output_shapes
:???????????0*
ksize
*
paddingSAME*
strides
?
)enformer/trunk/stem/max_pooling1d/SqueezeSqueeze2enformer/trunk/stem/max_pooling1d/MaxPool:output:0*
T0*-
_output_shapes
:???????????0*
squeeze_dims
?
fenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpoenformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:0*
dtype0?
[enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/IdentityIdentitynenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
henformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpqenformer_trunk_conv_tower_conv_tower_block_0_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:0*
dtype0?
]enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Identity_1Identitypenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
`enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/addAddV2fenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Identity_1:output:0ienformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:0?
`enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtbenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:0?
menformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOpvenformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:0*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mulMuldenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0uenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:0?
`enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul_1Mul2enformer/trunk/stem/max_pooling1d/Squeeze:output:0benformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????0?
`enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul_2Muldenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Identity:output:0benformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:0?
ienformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOprenformer_trunk_conv_tower_conv_tower_block_0_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:0*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/subSubqenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0denformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:0?
`enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2denformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0benformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????0?
=enformer/trunk/conv_tower/conv_tower_block_0/conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
;enformer/trunk/conv_tower/conv_tower_block_0/conv_block/mulMulFenformer/trunk/conv_tower/conv_tower_block_0/conv_block/mul/x:output:0denformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????0?
?enformer/trunk/conv_tower/conv_tower_block_0/conv_block/SigmoidSigmoid?enformer/trunk/conv_tower/conv_tower_block_0/conv_block/mul:z:0*
T0*-
_output_shapes
:???????????0?
=enformer/trunk/conv_tower/conv_tower_block_0/conv_block/mul_1MulCenformer/trunk/conv_tower/conv_tower_block_0/conv_block/Sigmoid:y:0denformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????0?
Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims
ExpandDimsAenformer/trunk/conv_tower/conv_tower_block_0/conv_block/mul_1:z:0cenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????0?
genformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOppenformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:0@*
dtype0?
\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Xenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsoenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0eenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0@?
Kenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolutionConv2D_enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims:output:0aenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Senformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/SqueezeSqueezeTenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims

??????????
Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOp_enformer_trunk_conv_tower_conv_tower_block_0_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Genformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAddBiasAdd\enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/Squeeze:output:0^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????@?
penformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpyenformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:@*
dtype0?
eenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/IdentityIdentityxenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
renformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOp{enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:@*
dtype0?
genformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Identity_1Identityzenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
henformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/addAddV2penformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Identity_1:output:0senformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtlenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:@?
wenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp?enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mulMulnenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1MulPenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd:output:0lenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????@?
jenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2Mulnenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Identity:output:0lenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:@?
senformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOp|enformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/subSub{enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0nenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2nenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0lenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????@?
Genformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
Eenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mulMulPenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mul/x:output:0nenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
Ienformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/SigmoidSigmoidIenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mul:z:0*
T0*-
_output_shapes
:???????????@?
Genformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/Sigmoid:y:0nenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims
ExpandDimsKenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/mul_1:z:0menformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
qenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpzenformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0?
fenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
benformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsyenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0oenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@?
Uenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolutionConv2Dienformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims:output:0kenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
]enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/SqueezeSqueeze^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims

??????????
`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpienformer_trunk_conv_tower_conv_tower_block_0_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Qenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAddBiasAddfenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/Squeeze:output:0henformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????@?
9enformer/trunk/conv_tower/conv_tower_block_0/residual/addAddV2Penformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd:output:0Zenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd:output:0*
T0*-
_output_shapes
:???????????@?
Kenformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Genformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/ExpandDims
ExpandDims=enformer/trunk/conv_tower/conv_tower_block_0/residual/add:z:0Tenformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
Denformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/MaxPoolMaxPoolPenformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/ExpandDims:output:0*1
_output_shapes
:???????????@*
ksize
*
paddingSAME*
strides
?
Denformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/SqueezeSqueezeMenformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/MaxPool:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims
?
fenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpoenformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:@*
dtype0?
[enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/IdentityIdentitynenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
henformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpqenformer_trunk_conv_tower_conv_tower_block_1_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:@*
dtype0?
]enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Identity_1Identitypenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/addAddV2fenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Identity_1:output:0ienformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtbenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:@?
menformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOpvenformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mulMuldenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0uenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_0/max_pooling1d_1/Squeeze:output:0benformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????@?
`enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul_2Muldenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Identity:output:0benformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:@?
ienformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOprenformer_trunk_conv_tower_conv_tower_block_1_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/subSubqenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0denformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2denformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0benformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????@?
=enformer/trunk/conv_tower/conv_tower_block_1/conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
;enformer/trunk/conv_tower/conv_tower_block_1/conv_block/mulMulFenformer/trunk/conv_tower/conv_tower_block_1/conv_block/mul/x:output:0denformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
?enformer/trunk/conv_tower/conv_tower_block_1/conv_block/SigmoidSigmoid?enformer/trunk/conv_tower/conv_tower_block_1/conv_block/mul:z:0*
T0*-
_output_shapes
:???????????@?
=enformer/trunk/conv_tower/conv_tower_block_1/conv_block/mul_1MulCenformer/trunk/conv_tower/conv_tower_block_1/conv_block/Sigmoid:y:0denformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims
ExpandDimsAenformer/trunk/conv_tower/conv_tower_block_1/conv_block/mul_1:z:0cenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
genformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOppenformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0?
\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Xenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsoenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0eenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@?
Kenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolutionConv2D_enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims:output:0aenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Senformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/SqueezeSqueezeTenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims

??????????
Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOp_enformer_trunk_conv_tower_conv_tower_block_1_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Genformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAddBiasAdd\enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/Squeeze:output:0^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????@?
penformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpyenformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:@*
dtype0?
eenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/IdentityIdentityxenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
renformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOp{enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:@*
dtype0?
genformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Identity_1Identityzenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
henformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/addAddV2penformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Identity_1:output:0senformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtlenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:@?
wenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp?enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mulMulnenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1MulPenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd:output:0lenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????@?
jenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2Mulnenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Identity:output:0lenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:@?
senformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOp|enformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/subSub{enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0nenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:@?
jenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2nenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0lenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????@?
Genformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
Eenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mulMulPenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mul/x:output:0nenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
Ienformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/SigmoidSigmoidIenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mul:z:0*
T0*-
_output_shapes
:???????????@?
Genformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/Sigmoid:y:0nenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims
ExpandDimsKenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/mul_1:z:0menformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
qenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpzenformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0?
fenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
benformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsyenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0oenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@?
Uenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolutionConv2Dienformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims:output:0kenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
]enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/SqueezeSqueeze^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims

??????????
`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpienformer_trunk_conv_tower_conv_tower_block_1_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Qenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAddBiasAddfenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/Squeeze:output:0henformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????@?
9enformer/trunk/conv_tower/conv_tower_block_1/residual/addAddV2Penformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd:output:0Zenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd:output:0*
T0*-
_output_shapes
:???????????@?
Kenformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Genformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/ExpandDims
ExpandDims=enformer/trunk/conv_tower/conv_tower_block_1/residual/add:z:0Tenformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
Denformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/MaxPoolMaxPoolPenformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/ExpandDims:output:0*1
_output_shapes
:???????????@*
ksize
*
paddingSAME*
strides
?
Denformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/SqueezeSqueezeMenformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/MaxPool:output:0*
T0*-
_output_shapes
:???????????@*
squeeze_dims
?
fenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpoenformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_read_readvariableop_resource*"
_output_shapes
:@*
dtype0?
[enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/IdentityIdentitynenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
henformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpqenformer_trunk_conv_tower_conv_tower_block_2_conv_block_exponential_moving_average_read_1_readvariableop_resource*"
_output_shapes
:@*
dtype0?
]enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Identity_1Identitypenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/addAddV2fenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Identity_1:output:0ienformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtbenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*"
_output_shapes
:@?
menformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOpvenformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mulMuldenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0uenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_1/max_pooling1d_2/Squeeze:output:0benformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:???????????@?
`enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul_2Muldenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Identity:output:0benformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*"
_output_shapes
:@?
ienformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOprenformer_trunk_conv_tower_conv_tower_block_2_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/subSubqenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0denformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*"
_output_shapes
:@?
`enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2denformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0benformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:???????????@?
=enformer/trunk/conv_tower/conv_tower_block_2/conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
;enformer/trunk/conv_tower/conv_tower_block_2/conv_block/mulMulFenformer/trunk/conv_tower/conv_tower_block_2/conv_block/mul/x:output:0denformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
?enformer/trunk/conv_tower/conv_tower_block_2/conv_block/SigmoidSigmoid?enformer/trunk/conv_tower/conv_tower_block_2/conv_block/mul:z:0*
T0*-
_output_shapes
:???????????@?
=enformer/trunk/conv_tower/conv_tower_block_2/conv_block/mul_1MulCenformer/trunk/conv_tower/conv_tower_block_2/conv_block/Sigmoid:y:0denformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*-
_output_shapes
:???????????@?
Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims
ExpandDimsAenformer/trunk/conv_tower/conv_tower_block_2/conv_block/mul_1:z:0cenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*1
_output_shapes
:???????????@?
genformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOppenformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0?
\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Xenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsoenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0eenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
Kenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolutionConv2D_enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims:output:0aenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
Senformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/SqueezeSqueezeTenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution:output:0*
T0*.
_output_shapes
:????????????*
squeeze_dims

??????????
Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOp_enformer_trunk_conv_tower_conv_tower_block_2_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Genformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAddBiasAdd\enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/Squeeze:output:0^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*.
_output_shapes
:?????????????
penformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpyenformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*#
_output_shapes
:?*
dtype0?
eenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/IdentityIdentityxenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
renformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOp{enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*#
_output_shapes
:?*
dtype0?
genformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Identity_1Identityzenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
henformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/addAddV2penformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Identity_1:output:0senformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtlenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*#
_output_shapes
:??
wenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp?enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mulMulnenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1MulPenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd:output:0lenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*.
_output_shapes
:?????????????
jenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2Mulnenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Identity:output:0lenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*#
_output_shapes
:??
senformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOp|enformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/subSub{enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0nenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2nenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0lenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*.
_output_shapes
:?????????????
Genformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
Eenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mulMulPenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mul/x:output:0nenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
Ienformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/SigmoidSigmoidIenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mul:z:0*
T0*.
_output_shapes
:?????????????
Genformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/Sigmoid:y:0nenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims
ExpandDimsKenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/mul_1:z:0menformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*2
_output_shapes 
:?????????????
qenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpzenformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0?
fenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
benformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsyenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0oenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
Uenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolutionConv2Dienformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims:output:0kenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
]enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/SqueezeSqueeze^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution:output:0*
T0*.
_output_shapes
:????????????*
squeeze_dims

??????????
`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpienformer_trunk_conv_tower_conv_tower_block_2_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Qenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAddBiasAddfenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/Squeeze:output:0henformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*.
_output_shapes
:?????????????
9enformer/trunk/conv_tower/conv_tower_block_2/residual/addAddV2Penformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd:output:0Zenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd:output:0*
T0*.
_output_shapes
:?????????????
Kenformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Genformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/ExpandDims
ExpandDims=enformer/trunk/conv_tower/conv_tower_block_2/residual/add:z:0Tenformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/ExpandDims/dim:output:0*
T0*2
_output_shapes 
:?????????????
Denformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/MaxPoolMaxPoolPenformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/ExpandDims:output:0*2
_output_shapes 
:????????????*
ksize
*
paddingSAME*
strides
?
Denformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/SqueezeSqueezeMenformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/MaxPool:output:0*
T0*.
_output_shapes
:????????????*
squeeze_dims
?
fenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpoenformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_read_readvariableop_resource*#
_output_shapes
:?*
dtype0?
[enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/IdentityIdentitynenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
henformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpqenformer_trunk_conv_tower_conv_tower_block_3_conv_block_exponential_moving_average_read_1_readvariableop_resource*#
_output_shapes
:?*
dtype0?
]enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Identity_1Identitypenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
`enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/addAddV2fenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Identity_1:output:0ienformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*#
_output_shapes
:??
`enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtbenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*#
_output_shapes
:??
menformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOpvenformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mulMuldenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0uenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
`enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_2/max_pooling1d_3/Squeeze:output:0benformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*.
_output_shapes
:?????????????
`enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul_2Muldenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Identity:output:0benformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*#
_output_shapes
:??
ienformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOprenformer_trunk_conv_tower_conv_tower_block_3_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/subSubqenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0denformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*#
_output_shapes
:??
`enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2denformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0benformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*.
_output_shapes
:?????????????
=enformer/trunk/conv_tower/conv_tower_block_3/conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
;enformer/trunk/conv_tower/conv_tower_block_3/conv_block/mulMulFenformer/trunk/conv_tower/conv_tower_block_3/conv_block/mul/x:output:0denformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
?enformer/trunk/conv_tower/conv_tower_block_3/conv_block/SigmoidSigmoid?enformer/trunk/conv_tower/conv_tower_block_3/conv_block/mul:z:0*
T0*.
_output_shapes
:?????????????
=enformer/trunk/conv_tower/conv_tower_block_3/conv_block/mul_1MulCenformer/trunk/conv_tower/conv_tower_block_3/conv_block/Sigmoid:y:0denformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims
ExpandDimsAenformer/trunk/conv_tower/conv_tower_block_3/conv_block/mul_1:z:0cenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*2
_output_shapes 
:?????????????
genformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOppenformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0?
\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Xenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsoenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0eenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
Kenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolutionConv2D_enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims:output:0aenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
Senformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/SqueezeSqueezeTenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution:output:0*
T0*.
_output_shapes
:????????????*
squeeze_dims

??????????
Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOp_enformer_trunk_conv_tower_conv_tower_block_3_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Genformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAddBiasAdd\enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/Squeeze:output:0^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*.
_output_shapes
:?????????????
penformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpyenformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*#
_output_shapes
:?*
dtype0?
eenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/IdentityIdentityxenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
renformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOp{enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*#
_output_shapes
:?*
dtype0?
genformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Identity_1Identityzenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
henformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/addAddV2penformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Identity_1:output:0senformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtlenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*#
_output_shapes
:??
wenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp?enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mulMulnenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1MulPenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd:output:0lenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*.
_output_shapes
:?????????????
jenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2Mulnenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Identity:output:0lenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*#
_output_shapes
:??
senformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOp|enformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
henformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/subSub{enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0nenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*#
_output_shapes
:??
jenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2nenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0lenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*.
_output_shapes
:?????????????
Genformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
Eenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mulMulPenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mul/x:output:0nenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
Ienformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/SigmoidSigmoidIenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mul:z:0*
T0*.
_output_shapes
:?????????????
Genformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/Sigmoid:y:0nenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*.
_output_shapes
:?????????????
denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims
ExpandDimsKenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/mul_1:z:0menformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*2
_output_shapes 
:?????????????
qenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpzenformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0?
fenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
benformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsyenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0oenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
Uenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolutionConv2Dienformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims:output:0kenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
]enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/SqueezeSqueeze^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution:output:0*
T0*.
_output_shapes
:????????????*
squeeze_dims

??????????
`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpienformer_trunk_conv_tower_conv_tower_block_3_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Qenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAddBiasAddfenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/Squeeze:output:0henformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*.
_output_shapes
:?????????????
9enformer/trunk/conv_tower/conv_tower_block_3/residual/addAddV2Penformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd:output:0Zenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd:output:0*
T0*.
_output_shapes
:?????????????
Kenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Genformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/ExpandDims
ExpandDims=enformer/trunk/conv_tower/conv_tower_block_3/residual/add:z:0Tenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/ExpandDims/dim:output:0*
T0*2
_output_shapes 
:?????????????
Denformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/MaxPoolMaxPoolPenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/ExpandDims:output:0*1
_output_shapes
:??????????`?*
ksize
*
paddingSAME*
strides
?
Denformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/SqueezeSqueezeMenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/MaxPool:output:0*
T0*-
_output_shapes
:??????????`?*
squeeze_dims
?
\enformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/meanMeanMenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/Squeeze:output:0eenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/SquaredDifferenceSquaredDifferenceMenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/Squeeze:output:0[enformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_0_mha_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul_1MulMenformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/Squeeze:output:0Oenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_0/mha/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_0_mha_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/ShapeShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_sliceStridedSliceYenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stack:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stack_1:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/ProdProdaenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice:output:0Yenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1StridedSliceYenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stack:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stack_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concatConcatV2Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Prod:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/ReshapeReshapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_0_mha_attention_0_q_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMulMatMul[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Reshape:output:0denformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape_1ShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2StridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape_1:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stack:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stack_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape_2ShapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul:product:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3StridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Shape_2:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stack:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stack_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat_1ConcatV2cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_2:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/strided_slice_3:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul:product:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/ShapeShape]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Reshape_1:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_sliceStridedSliceUenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/Shape:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stack:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stack_1:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concatConcatV2]enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/strided_slice:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concat/values_1:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/ReshapeReshape]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply/Reshape_1:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose	TransposeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape/Reshape:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/ShapeShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/ProdProdcenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concatConcatV2Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Prod:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/ReshapeReshapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_0_mha_attention_0_k_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Reshape:output:0denformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape_1ShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape_2ShapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Shape_2:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/ShapeShape_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/Shape:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concatConcatV2_enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/strided_slice:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/ReshapeReshape_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_1/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_1	TransposeYenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_1/Reshape:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/ShapeShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/ProdProdcenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concatConcatV2Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Prod:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/ReshapeReshapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_0_mha_attention_0_v_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Reshape:output:0denformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape_1ShapeQenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape_1:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape_2ShapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Shape_2:output:0kenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/ShapeShape_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/Shape:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0        ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concatConcatV2_enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/strided_slice:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/ReshapeReshape_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_2/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/concat:output:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_2	TransposeYenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_2/Reshape:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_2/perm:output:0*
T0*0
_output_shapes
:??????????`?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/mulMulLenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose:y:0Menformer/trunk/transformer/transformer_block_0/mha/attention_0/mul/y:output:0*
T0*0
_output_shapes
:??????????`@?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/range/startConst*
_output_shapes
: *
dtype0*
valueB
 * ????
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/rangeRangeSenformer/trunk/transformer/transformer_block_0/mha/attention_0/range/start:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/range/limit:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/range/delta:output:0*

Tidx0*
_output_shapes

:???
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_sliceStridedSliceMenformer/trunk/transformer/transformer_block_0/mha/attention_0/range:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stack:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stack_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*
new_axis_mask?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/AbsAbsUenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice:output:0*
T0* 
_output_shapes
:
???
Eenformer/trunk/transformer/transformer_block_0/mha/attention_0/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?`?
Cenformer/trunk/transformer/transformer_block_0/mha/attention_0/CastCastNenformer/trunk/transformer/transformer_block_0/mha/attention_0/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/LogLogGenformer/trunk/transformer/transformer_block_0/mha/attention_0/Cast:y:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_1LogOenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_1/x:output:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/truedivRealDivFenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log:y:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_1:y:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @@?
Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/numConst*
_output_shapes
: *
dtype0*
value	B :?
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/CastCastTenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast_1CastPenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastArgsBroadcastArgsVenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_1:output:0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastToBroadcastToVenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/start:output:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastTo_1BroadcastToJenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv:z:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims
ExpandDims\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastTo:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims_1
ExpandDims^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/BroadcastTo_1:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_sliceStridedSliceXenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_3:output:0denformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stack:output:0fenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stack_1:output:0fenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/addAddV2^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/add/y:output:0*
T0*
_output_shapes
: ?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2SelectV2cenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2/condition:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2/t:output:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/add:z:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/subSubPenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast:y:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/MaximumMaximumOenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub:z:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_1SubPenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_1/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum_1MaximumQenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_1:z:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_2Sub]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast_2CastUenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/truedivRealDivQenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/sub_2:z:0Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast_2:y:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/GreaterEqualGreaterEqualPenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast:y:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_1SelectV2Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/GreaterEqual:z:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum_1:z:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/CastCast[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/rangeRange\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/start:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/Cast:y:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast_3CastVenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1Range^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1/start:output:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/strided_slice:output:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1/delta:output:0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/EqualEqualYenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/range_1:output:0*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_2SelectV2Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Equal:z:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Maximum:z:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ReshapeReshapeRenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast_3:y:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_2:output:0*
T0*
_output_shapes
:?
Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/mulMulSenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/truediv:z:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Reshape:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/add_1AddV2[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims:output:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/mul:z:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/concatConcatV2[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/add_1:z:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/ExpandDims_1:output:0Yenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_3SelectV2Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Equal:z:0Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Shape_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SliceSliceWenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/concat:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/zeros_like:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/PowPowMenformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow/x:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace/Slice:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_3ReshapeFenformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_3/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_1AbsFenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs:y:0*
T0* 
_output_shapes
:
???
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_2LogOenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_2/x:output:0*
T0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/NegNegHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_2:y:0*
T0*
_output_shapes
: ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_1RealDivFenformer/trunk/transformer/transformer_block_0/mha/attention_0/Neg:y:0Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_3:output:0*
T0*"
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1StridedSliceHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_1:y:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stack:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stack_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_1MulLenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_1:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_1:output:0*
T0*$
_output_shapes
:???
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/ExpExpHenformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_1:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_2AbsUenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice:output:0*
T0* 
_output_shapes
:
???
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@?
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1RangeUenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/start:output:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/limit:output:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1/delta:output:0*

Tidx0*
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow_1PowOenformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow_1/x:output:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/range_1:output:0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/subSubHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Pow_1:z:0Menformer/trunk/transformer/transformer_block_0/mha/attention_0/sub/y:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_4ReshapeFenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_4/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_3AbsHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_2:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2StridedSliceHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_3:y:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stack:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stack_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/GreaterGreaterQenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_4:output:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_2:output:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_0/mha/attention_0/Cast_1CastJenformer/trunk/transformer/transformer_block_0/mha/attention_0/Greater:z:0*

DstT0*

SrcT0
*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_4AbsUenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice:output:0*
T0* 
_output_shapes
:
???
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @E?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/stopConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/CastCastVenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast_1CastRenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastArgsBroadcastArgsXenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape:output:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_1:output:0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastToBroadcastToXenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/start:output:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastTo_1BroadcastToWenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/stop:output:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims
ExpandDims^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastTo:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims_1
ExpandDims`enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/BroadcastTo_1:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_sliceStridedSliceZenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_3:output:0fenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stack:output:0henformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stack_1:output:0henformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/addAddV2`enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/add/y:output:0*
T0*
_output_shapes
: ?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2SelectV2eenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2/condition:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2/t:output:0Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/add:z:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/subSubRenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast:y:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/MaximumMaximumQenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub:z:0\enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_1SubRenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum_1MaximumSenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_1:z:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_2Sub_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast_2CastWenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/truedivRealDivSenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/sub_2:z:0Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/GreaterEqualGreaterEqualRenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast:y:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_1SelectV2Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/GreaterEqual:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum_1:z:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/CastCast]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/rangeRange^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/start:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/Cast:y:0^enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast_3CastXenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1Range`enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1/start:output:0`enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/strided_slice:output:0`enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1/delta:output:0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/EqualEqual[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2:output:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/range_1:output:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_2SelectV2Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Equal:z:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Maximum:z:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ReshapeReshapeTenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast_3:y:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/mulMulUenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/truediv:z:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Reshape:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/add_1AddV2]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/mul:z:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/concatConcatV2]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/add_1:z:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_3SelectV2Senformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Equal:z:0Renformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SliceSliceYenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/concat:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/zeros_like:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_5ReshapeXenformer/trunk/transformer/transformer_block_0/mha/attention_0/linspace_1/Slice:output:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_5/shape:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?D?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_2RealDivQenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_2/y:output:0*
T0*"
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2PowLenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_2:z:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2/y:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  J?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_3RealDivQenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_3/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_5AbsHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_4:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3StridedSliceHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Abs_5:y:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stack:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stack_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_1SubHenformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2:z:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_1/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/XlogyXlogyHenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_1:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_2MulLenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_3:z:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_2SubHenformer/trunk/transformer/transformer_block_0/mha/attention_0/Xlogy:z:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_2:z:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_0/mha/attention_0/LgammaLgammaHenformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_3LogLenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_3MulHenformer/trunk/transformer/transformer_block_0/mha/attention_0/pow_2:z:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Log_3:y:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_3SubIenformer/trunk/transformer/transformer_block_0/mha/attention_0/Lgamma:y:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_4SubHenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_2:z:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_3:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/Exp_1ExpHenformer/trunk/transformer/transformer_block_0/mha/attention_0/sub_4:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/addAddV2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Exp_1:y:0Menformer/trunk/transformer/transformer_block_0/mha/attention_0/add/y:output:0*
T0*$
_output_shapes
:???
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Benformer/trunk/transformer/transformer_block_0/mha/attention_0/MaxMaxFenformer/trunk/transformer/transformer_block_0/mha/attention_0/add:z:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/Max/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_4RealDivFenformer/trunk/transformer/transformer_block_0/mha/attention_0/add:z:0Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/Max:output:0*
T0*$
_output_shapes
:???
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Eenformer/trunk/transformer/transformer_block_0/mha/attention_0/concatConcatV2Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/Exp:y:0Ienformer/trunk/transformer/transformer_block_0/mha/attention_0/Cast_1:y:0Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/truediv_4:z:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/concat/axis:output:0*
N*
T0*$
_output_shapes
:???
Cenformer/trunk/transformer/transformer_block_0/mha/attention_0/SignSignUenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice:output:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4StridedSliceGenformer/trunk/transformer/transformer_block_0/mha/attention_0/Sign:y:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stack:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stack_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_4MulWenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_4:output:0Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat:output:0*
T0*$
_output_shapes
:???
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_1ConcatV2Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat:output:0Henformer/trunk/transformer/transformer_block_0/mha/attention_0/mul_4:z:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_1/axis:output:0*
N*
T0*$
_output_shapes
:???
Zenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?_     ?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/ReshapeReshapePenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_1:output:0cenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape/shape:output:0*
T0* 
_output_shapes
:
???
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOpReadVariableOpgenformer_trunk_transformer_transformer_block_0_mha_attention_0_r_k_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape:output:0fenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOp:value:0*
T0*!
_output_shapes
:????
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?_     ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape_1ReshapeYenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul:product:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape_1/shape:output:0*
T0*%
_output_shapes
:????
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?_     @   ?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_6/ReshapeReshape_enformer/trunk/transformer/transformer_block_0/mha/attention_0/batch_apply_3/Reshape_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:??@?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_3	TransposeYenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_6/Reshape:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_3/perm:output:0*
T0*(
_output_shapes
:??@?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_0_mha_attention_0_add_1_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1AddV2Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/mul:z:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Eenformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMulBatchMatMulV2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1:z:0Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_1:y:0*
T0*1
_output_shapes
:??????????`?`*
adj_y(?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_0_mha_attention_0_add_2_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2AddV2Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/mul:z:0[enformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul_1BatchMatMulV2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2:z:0Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_3:y:0*
T0*2
_output_shapes 
:??????????`??*
adj_y(?
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5StridedSlicePenformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stack:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stack_1:output:0_enformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????`*

begin_mask*
ellipsis_mask?
Ienformer/trunk/transformer/transformer_block_0/mha/attention_0/zeros_like	ZerosLikeWenformer/trunk/transformer/transformer_block_0/mha/attention_0/strided_slice_5:output:0*
T0*0
_output_shapes
:??????????`?
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_2ConcatV2Menformer/trunk/transformer/transformer_block_0/mha/attention_0/zeros_like:y:0Penformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul_1:output:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_2/axis:output:0*
N*
T0*2
_output_shapes 
:??????????`???
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    `   0  ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_7ReshapePenformer/trunk/transformer/transformer_block_0/mha/attention_0/concat_2:output:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_7/shape:output:0*
T0*2
_output_shapes 
:????????????`?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"               ?
Ienformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/SliceSliceQenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_7:output:0Senformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice/begin:output:0Renformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice/size:output:0*
Index0*
T0*2
_output_shapes 
:????????????`?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    0  ?_  ?
Henformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_8ReshapeMenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice:output:0Wenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_8/shape:output:0*
T0*2
_output_shapes 
:??????????`???
Lenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                ?
Kenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"???????????? 0  ?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1SliceQenformer/trunk/transformer/transformer_block_0/mha/attention_0/Reshape_8:output:0Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1/begin:output:0Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1/size:output:0*
Index0*
T0*1
_output_shapes
:??????????`?`?
Denformer/trunk/transformer/transformer_block_0/mha/attention_0/add_3AddV2Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul:output:0Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/Slice_1:output:0*
T0*1
_output_shapes
:??????????`?`?
Fenformer/trunk/transformer/transformer_block_0/mha/attention_0/SoftmaxSoftmaxHenformer/trunk/transformer/transformer_block_0/mha/attention_0/add_3:z:0*
T0*1
_output_shapes
:??????????`?`?
Genformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul_2BatchMatMulV2Penformer/trunk/transformer/transformer_block_0/mha/attention_0/Softmax:softmax:0Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_2:y:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_4	TransposePenformer/trunk/transformer/transformer_block_0/mha/attention_0/MatMul_2:output:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_4/perm:output:0*
T0*0
_output_shapes
:??????????`?
Nenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/ShapeShapeNenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_4:y:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/Shape:output:0eenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:??
Tenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concatConcatV2_enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/strided_slice:output:0aenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/ReshapeReshapeNenformer/trunk/transformer/transformer_block_0/mha/attention_0/transpose_4:y:0Xenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/concat:output:0*
T0*-
_output_shapes
:??????????`??
denformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOpReadVariableOpmenformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Uenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMulBatchMatMulV2Yenformer/trunk/transformer/transformer_block_0/mha/attention_0/reshape_9/Reshape:output:0lenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
aenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOpReadVariableOpjenformer_trunk_transformer_transformer_block_0_mha_attention_0_embedding_layer_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Renformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/AddAddV2^enformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul:output:0ienformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
;enformer/trunk/transformer/transformer_block_0/residual/addAddV2Menformer/trunk/conv_tower/conv_tower_block_3/max_pooling1d_4/Squeeze:output:0Venformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/meanMean?enformer/trunk/transformer/transformer_block_0/residual/add:z:0eenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/SquaredDifferenceSquaredDifference?enformer/trunk/transformer/transformer_block_0/residual/add:z:0[enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_0_mlp_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul_1Mul?enformer/trunk/transformer/transformer_block_0/residual/add:z:0Oenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_0_mlp_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Oenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOpReadVariableOpXenformer_trunk_transformer_transformer_block_0_mlp_linear_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
@enformer/trunk/transformer/transformer_block_0/mlp/linear/MatMulBatchMatMulV2Qenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/add_1:z:0Wenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOpReadVariableOpUenformer_trunk_transformer_transformer_block_0_mlp_linear_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
=enformer/trunk/transformer/transformer_block_0/mlp/linear/AddAddV2Ienformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul:output:0Tenformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
7enformer/trunk/transformer/transformer_block_0/mlp/ReluReluAenformer/trunk/transformer/transformer_block_0/mlp/linear/Add:z:0*
T0*-
_output_shapes
:??????????`??
Qenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOpReadVariableOpZenformer_trunk_transformer_transformer_block_0_mlp_linear_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Benformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1BatchMatMulV2Eenformer/trunk/transformer/transformer_block_0/mlp/Relu:activations:0Yenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOpReadVariableOpWenformer_trunk_transformer_transformer_block_0_mlp_linear_add_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
?enformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1AddV2Kenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1:output:0Venformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
=enformer/trunk/transformer/transformer_block_0/residual/add_1AddV2?enformer/trunk/transformer/transformer_block_0/residual/add:z:0Cenformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/meanMeanAenformer/trunk/transformer/transformer_block_0/residual/add_1:z:0eenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/SquaredDifferenceSquaredDifferenceAenformer/trunk/transformer/transformer_block_0/residual/add_1:z:0[enformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_1_mha_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul_1MulAenformer/trunk/transformer/transformer_block_0/residual/add_1:z:0Oenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_1/mha/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_1_mha_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/ShapeShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_sliceStridedSliceYenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stack:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stack_1:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/ProdProdaenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice:output:0Yenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1StridedSliceYenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stack:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stack_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concatConcatV2Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Prod:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/ReshapeReshapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_1_mha_attention_1_q_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMulMatMul[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Reshape:output:0denformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape_1ShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2StridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape_1:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stack:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stack_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape_2ShapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul:product:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3StridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Shape_2:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stack:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stack_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat_1ConcatV2cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_2:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/strided_slice_3:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul:product:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/ShapeShape]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Reshape_1:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_sliceStridedSliceUenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/Shape:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stack:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stack_1:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concatConcatV2]enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/strided_slice:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concat/values_1:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/ReshapeReshape]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply/Reshape_1:output:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose	TransposeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape/Reshape:output:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/ShapeShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/ProdProdcenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concatConcatV2Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Prod:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/ReshapeReshapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_1_mha_attention_1_k_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Reshape:output:0denformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape_1ShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape_2ShapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Shape_2:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/ShapeShape_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/Shape:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concatConcatV2_enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/strided_slice:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/ReshapeReshape_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_1/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_1	TransposeYenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_1/Reshape:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/ShapeShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/ProdProdcenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concatConcatV2Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Prod:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/ReshapeReshapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_1_mha_attention_1_v_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Reshape:output:0denformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape_1ShapeQenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape_1:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape_2ShapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Shape_2:output:0kenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/ShapeShape_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/Shape:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0        ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concatConcatV2_enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/strided_slice:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/ReshapeReshape_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_2/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/concat:output:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_2	TransposeYenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_2/Reshape:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_2/perm:output:0*
T0*0
_output_shapes
:??????????`?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/mulMulLenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose:y:0Menformer/trunk/transformer/transformer_block_1/mha/attention_1/mul/y:output:0*
T0*0
_output_shapes
:??????????`@?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/range/startConst*
_output_shapes
: *
dtype0*
valueB
 * ????
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/rangeRangeSenformer/trunk/transformer/transformer_block_1/mha/attention_1/range/start:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/range/limit:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/range/delta:output:0*

Tidx0*
_output_shapes

:???
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_sliceStridedSliceMenformer/trunk/transformer/transformer_block_1/mha/attention_1/range:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stack:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stack_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*
new_axis_mask?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/AbsAbsUenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice:output:0*
T0* 
_output_shapes
:
???
Eenformer/trunk/transformer/transformer_block_1/mha/attention_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?`?
Cenformer/trunk/transformer/transformer_block_1/mha/attention_1/CastCastNenformer/trunk/transformer/transformer_block_1/mha/attention_1/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/LogLogGenformer/trunk/transformer/transformer_block_1/mha/attention_1/Cast:y:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_1LogOenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_1/x:output:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/truedivRealDivFenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log:y:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_1:y:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @@?
Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/numConst*
_output_shapes
: *
dtype0*
value	B :?
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/CastCastTenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast_1CastPenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastArgsBroadcastArgsVenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_1:output:0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastToBroadcastToVenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/start:output:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastTo_1BroadcastToJenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv:z:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims
ExpandDims\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastTo:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims_1
ExpandDims^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/BroadcastTo_1:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_sliceStridedSliceXenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_3:output:0denformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stack:output:0fenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stack_1:output:0fenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/addAddV2^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice:output:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/add/y:output:0*
T0*
_output_shapes
: ?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2SelectV2cenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2/condition:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2/t:output:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/add:z:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/subSubPenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast:y:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/MaximumMaximumOenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub:z:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_1SubPenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_1/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum_1MaximumQenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_1:z:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_2Sub]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast_2CastUenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/truedivRealDivQenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/sub_2:z:0Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast_2:y:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/GreaterEqualGreaterEqualPenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast:y:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_1SelectV2Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/GreaterEqual:z:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum_1:z:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/CastCast[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/rangeRange\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/start:output:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/Cast:y:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast_3CastVenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1Range^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1/start:output:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/strided_slice:output:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1/delta:output:0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/EqualEqualYenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/range_1:output:0*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_2SelectV2Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Equal:z:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Maximum:z:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ReshapeReshapeRenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast_3:y:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_2:output:0*
T0*
_output_shapes
:?
Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/mulMulSenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/truediv:z:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Reshape:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/add_1AddV2[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims:output:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/mul:z:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/concatConcatV2[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/add_1:z:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/ExpandDims_1:output:0Yenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_3SelectV2Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Equal:z:0Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Shape_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SliceSliceWenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/concat:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/zeros_like:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/PowPowMenformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow/x:output:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace/Slice:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_3ReshapeFenformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_3/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_1AbsFenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs:y:0*
T0* 
_output_shapes
:
???
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_2LogOenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_2/x:output:0*
T0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/NegNegHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_2:y:0*
T0*
_output_shapes
: ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_1RealDivFenformer/trunk/transformer/transformer_block_1/mha/attention_1/Neg:y:0Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_3:output:0*
T0*"
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1StridedSliceHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_1:y:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stack:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stack_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_1MulLenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_1:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_1:output:0*
T0*$
_output_shapes
:???
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/ExpExpHenformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_1:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_2AbsUenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice:output:0*
T0* 
_output_shapes
:
???
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@?
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1RangeUenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/start:output:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/limit:output:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1/delta:output:0*

Tidx0*
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow_1PowOenformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow_1/x:output:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/range_1:output:0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/subSubHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Pow_1:z:0Menformer/trunk/transformer/transformer_block_1/mha/attention_1/sub/y:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_4ReshapeFenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_4/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_3AbsHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_2:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2StridedSliceHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_3:y:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stack:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stack_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/GreaterGreaterQenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_4:output:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_2:output:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_1/mha/attention_1/Cast_1CastJenformer/trunk/transformer/transformer_block_1/mha/attention_1/Greater:z:0*

DstT0*

SrcT0
*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_4AbsUenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice:output:0*
T0* 
_output_shapes
:
???
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @E?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/stopConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/CastCastVenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast_1CastRenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastArgsBroadcastArgsXenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape:output:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_1:output:0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastToBroadcastToXenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/start:output:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastTo_1BroadcastToWenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/stop:output:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims
ExpandDims^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastTo:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims_1
ExpandDims`enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/BroadcastTo_1:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_sliceStridedSliceZenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_3:output:0fenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stack:output:0henformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stack_1:output:0henformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/addAddV2`enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/add/y:output:0*
T0*
_output_shapes
: ?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2SelectV2eenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2/condition:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2/t:output:0Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/add:z:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/subSubRenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast:y:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/MaximumMaximumQenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub:z:0\enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_1SubRenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum_1MaximumSenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_1:z:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_2Sub_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast_2CastWenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/truedivRealDivSenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/sub_2:z:0Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/GreaterEqualGreaterEqualRenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast:y:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_1SelectV2Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/GreaterEqual:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum_1:z:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/CastCast]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/rangeRange^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/start:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/Cast:y:0^enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast_3CastXenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1Range`enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1/start:output:0`enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/strided_slice:output:0`enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1/delta:output:0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/EqualEqual[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2:output:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/range_1:output:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_2SelectV2Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Equal:z:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Maximum:z:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ReshapeReshapeTenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast_3:y:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/mulMulUenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/truediv:z:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Reshape:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/add_1AddV2]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/mul:z:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/concatConcatV2]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/add_1:z:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_3SelectV2Senformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Equal:z:0Renformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SliceSliceYenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/concat:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/zeros_like:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_5ReshapeXenformer/trunk/transformer/transformer_block_1/mha/attention_1/linspace_1/Slice:output:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_5/shape:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?D?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_2RealDivQenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_2/y:output:0*
T0*"
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2PowLenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_2:z:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2/y:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  J?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_3RealDivQenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_3/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_5AbsHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_4:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3StridedSliceHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Abs_5:y:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stack:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stack_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_1SubHenformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2:z:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_1/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/XlogyXlogyHenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_1:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_2MulLenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_3:z:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_2SubHenformer/trunk/transformer/transformer_block_1/mha/attention_1/Xlogy:z:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_2:z:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_1/mha/attention_1/LgammaLgammaHenformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_3LogLenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_3MulHenformer/trunk/transformer/transformer_block_1/mha/attention_1/pow_2:z:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Log_3:y:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_3SubIenformer/trunk/transformer/transformer_block_1/mha/attention_1/Lgamma:y:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_4SubHenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_2:z:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_3:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/Exp_1ExpHenformer/trunk/transformer/transformer_block_1/mha/attention_1/sub_4:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/addAddV2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Exp_1:y:0Menformer/trunk/transformer/transformer_block_1/mha/attention_1/add/y:output:0*
T0*$
_output_shapes
:???
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Benformer/trunk/transformer/transformer_block_1/mha/attention_1/MaxMaxFenformer/trunk/transformer/transformer_block_1/mha/attention_1/add:z:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/Max/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_4RealDivFenformer/trunk/transformer/transformer_block_1/mha/attention_1/add:z:0Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/Max:output:0*
T0*$
_output_shapes
:???
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Eenformer/trunk/transformer/transformer_block_1/mha/attention_1/concatConcatV2Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/Exp:y:0Ienformer/trunk/transformer/transformer_block_1/mha/attention_1/Cast_1:y:0Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/truediv_4:z:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/concat/axis:output:0*
N*
T0*$
_output_shapes
:???
Cenformer/trunk/transformer/transformer_block_1/mha/attention_1/SignSignUenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice:output:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4StridedSliceGenformer/trunk/transformer/transformer_block_1/mha/attention_1/Sign:y:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stack:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stack_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_4MulWenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_4:output:0Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat:output:0*
T0*$
_output_shapes
:???
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_1ConcatV2Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat:output:0Henformer/trunk/transformer/transformer_block_1/mha/attention_1/mul_4:z:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_1/axis:output:0*
N*
T0*$
_output_shapes
:???
Zenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?_     ?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/ReshapeReshapePenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_1:output:0cenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape/shape:output:0*
T0* 
_output_shapes
:
???
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOpReadVariableOpgenformer_trunk_transformer_transformer_block_1_mha_attention_1_r_k_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape:output:0fenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOp:value:0*
T0*!
_output_shapes
:????
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?_     ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape_1ReshapeYenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul:product:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape_1/shape:output:0*
T0*%
_output_shapes
:????
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?_     @   ?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_6/ReshapeReshape_enformer/trunk/transformer/transformer_block_1/mha/attention_1/batch_apply_3/Reshape_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:??@?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_3	TransposeYenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_6/Reshape:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_3/perm:output:0*
T0*(
_output_shapes
:??@?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_1_mha_attention_1_add_1_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1AddV2Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/mul:z:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Eenformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMulBatchMatMulV2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1:z:0Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_1:y:0*
T0*1
_output_shapes
:??????????`?`*
adj_y(?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_1_mha_attention_1_add_2_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2AddV2Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/mul:z:0[enformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul_1BatchMatMulV2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2:z:0Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_3:y:0*
T0*2
_output_shapes 
:??????????`??*
adj_y(?
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5StridedSlicePenformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stack:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stack_1:output:0_enformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????`*

begin_mask*
ellipsis_mask?
Ienformer/trunk/transformer/transformer_block_1/mha/attention_1/zeros_like	ZerosLikeWenformer/trunk/transformer/transformer_block_1/mha/attention_1/strided_slice_5:output:0*
T0*0
_output_shapes
:??????????`?
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_2ConcatV2Menformer/trunk/transformer/transformer_block_1/mha/attention_1/zeros_like:y:0Penformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul_1:output:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_2/axis:output:0*
N*
T0*2
_output_shapes 
:??????????`???
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    `   0  ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_7ReshapePenformer/trunk/transformer/transformer_block_1/mha/attention_1/concat_2:output:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_7/shape:output:0*
T0*2
_output_shapes 
:????????????`?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"               ?
Ienformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/SliceSliceQenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_7:output:0Senformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice/begin:output:0Renformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice/size:output:0*
Index0*
T0*2
_output_shapes 
:????????????`?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    0  ?_  ?
Henformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_8ReshapeMenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice:output:0Wenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_8/shape:output:0*
T0*2
_output_shapes 
:??????????`???
Lenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                ?
Kenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"???????????? 0  ?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1SliceQenformer/trunk/transformer/transformer_block_1/mha/attention_1/Reshape_8:output:0Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1/begin:output:0Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1/size:output:0*
Index0*
T0*1
_output_shapes
:??????????`?`?
Denformer/trunk/transformer/transformer_block_1/mha/attention_1/add_3AddV2Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul:output:0Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/Slice_1:output:0*
T0*1
_output_shapes
:??????????`?`?
Fenformer/trunk/transformer/transformer_block_1/mha/attention_1/SoftmaxSoftmaxHenformer/trunk/transformer/transformer_block_1/mha/attention_1/add_3:z:0*
T0*1
_output_shapes
:??????????`?`?
Genformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul_2BatchMatMulV2Penformer/trunk/transformer/transformer_block_1/mha/attention_1/Softmax:softmax:0Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_2:y:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_4	TransposePenformer/trunk/transformer/transformer_block_1/mha/attention_1/MatMul_2:output:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_4/perm:output:0*
T0*0
_output_shapes
:??????????`?
Nenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/ShapeShapeNenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_4:y:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/Shape:output:0eenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:??
Tenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concatConcatV2_enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/strided_slice:output:0aenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/ReshapeReshapeNenformer/trunk/transformer/transformer_block_1/mha/attention_1/transpose_4:y:0Xenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/concat:output:0*
T0*-
_output_shapes
:??????????`??
denformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOpReadVariableOpmenformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Uenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMulBatchMatMulV2Yenformer/trunk/transformer/transformer_block_1/mha/attention_1/reshape_9/Reshape:output:0lenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
aenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOpReadVariableOpjenformer_trunk_transformer_transformer_block_1_mha_attention_1_embedding_layer_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Renformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/AddAddV2^enformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul:output:0ienformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
;enformer/trunk/transformer/transformer_block_1/residual/addAddV2Aenformer/trunk/transformer/transformer_block_0/residual/add_1:z:0Venformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/meanMean?enformer/trunk/transformer/transformer_block_1/residual/add:z:0eenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/SquaredDifferenceSquaredDifference?enformer/trunk/transformer/transformer_block_1/residual/add:z:0[enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_1_mlp_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul_1Mul?enformer/trunk/transformer/transformer_block_1/residual/add:z:0Oenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_1_mlp_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Oenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOpReadVariableOpXenformer_trunk_transformer_transformer_block_1_mlp_linear_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
@enformer/trunk/transformer/transformer_block_1/mlp/linear/MatMulBatchMatMulV2Qenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/add_1:z:0Wenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOpReadVariableOpUenformer_trunk_transformer_transformer_block_1_mlp_linear_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
=enformer/trunk/transformer/transformer_block_1/mlp/linear/AddAddV2Ienformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul:output:0Tenformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
7enformer/trunk/transformer/transformer_block_1/mlp/ReluReluAenformer/trunk/transformer/transformer_block_1/mlp/linear/Add:z:0*
T0*-
_output_shapes
:??????????`??
Qenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOpReadVariableOpZenformer_trunk_transformer_transformer_block_1_mlp_linear_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Benformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1BatchMatMulV2Eenformer/trunk/transformer/transformer_block_1/mlp/Relu:activations:0Yenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOpReadVariableOpWenformer_trunk_transformer_transformer_block_1_mlp_linear_add_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
?enformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1AddV2Kenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1:output:0Venformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
=enformer/trunk/transformer/transformer_block_1/residual/add_1AddV2?enformer/trunk/transformer/transformer_block_1/residual/add:z:0Cenformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/meanMeanAenformer/trunk/transformer/transformer_block_1/residual/add_1:z:0eenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/SquaredDifferenceSquaredDifferenceAenformer/trunk/transformer/transformer_block_1/residual/add_1:z:0[enformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_2_mha_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul_1MulAenformer/trunk/transformer/transformer_block_1/residual/add_1:z:0Oenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_2/mha/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_2_mha_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/ShapeShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_sliceStridedSliceYenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stack:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stack_1:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/ProdProdaenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice:output:0Yenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1StridedSliceYenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stack:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stack_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concatConcatV2Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Prod:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/ReshapeReshapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_2_mha_attention_2_q_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMulMatMul[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Reshape:output:0denformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape_1ShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2StridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape_1:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stack:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stack_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape_2ShapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul:product:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3StridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Shape_2:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stack:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stack_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat_1ConcatV2cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_2:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/strided_slice_3:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul:product:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/ShapeShape]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Reshape_1:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_sliceStridedSliceUenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/Shape:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stack:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stack_1:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concatConcatV2]enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/strided_slice:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concat/values_1:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/ReshapeReshape]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply/Reshape_1:output:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose	TransposeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape/Reshape:output:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/ShapeShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/ProdProdcenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concatConcatV2Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Prod:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/ReshapeReshapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_2_mha_attention_2_k_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Reshape:output:0denformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape_1ShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape_2ShapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Shape_2:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/ShapeShape_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/Shape:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concatConcatV2_enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/strided_slice:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/ReshapeReshape_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_1/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_1	TransposeYenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_1/Reshape:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/ShapeShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/ProdProdcenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concatConcatV2Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Prod:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/ReshapeReshapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_2_mha_attention_2_v_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Reshape:output:0denformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape_1ShapeQenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape_1:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape_2ShapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Shape_2:output:0kenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/ShapeShape_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/Shape:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0        ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concatConcatV2_enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/strided_slice:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/ReshapeReshape_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_2/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/concat:output:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_2	TransposeYenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_2/Reshape:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_2/perm:output:0*
T0*0
_output_shapes
:??????????`?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/mulMulLenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose:y:0Menformer/trunk/transformer/transformer_block_2/mha/attention_2/mul/y:output:0*
T0*0
_output_shapes
:??????????`@?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/range/startConst*
_output_shapes
: *
dtype0*
valueB
 * ????
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/rangeRangeSenformer/trunk/transformer/transformer_block_2/mha/attention_2/range/start:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/range/limit:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/range/delta:output:0*

Tidx0*
_output_shapes

:???
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_sliceStridedSliceMenformer/trunk/transformer/transformer_block_2/mha/attention_2/range:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stack:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stack_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*
new_axis_mask?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/AbsAbsUenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice:output:0*
T0* 
_output_shapes
:
???
Eenformer/trunk/transformer/transformer_block_2/mha/attention_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?`?
Cenformer/trunk/transformer/transformer_block_2/mha/attention_2/CastCastNenformer/trunk/transformer/transformer_block_2/mha/attention_2/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/LogLogGenformer/trunk/transformer/transformer_block_2/mha/attention_2/Cast:y:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_1LogOenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_1/x:output:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/truedivRealDivFenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log:y:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_1:y:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @@?
Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/numConst*
_output_shapes
: *
dtype0*
value	B :?
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/CastCastTenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast_1CastPenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastArgsBroadcastArgsVenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_1:output:0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastToBroadcastToVenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/start:output:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastTo_1BroadcastToJenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv:z:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims
ExpandDims\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastTo:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims_1
ExpandDims^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/BroadcastTo_1:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_sliceStridedSliceXenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_3:output:0denformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stack:output:0fenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stack_1:output:0fenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/addAddV2^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice:output:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/add/y:output:0*
T0*
_output_shapes
: ?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2SelectV2cenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2/condition:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2/t:output:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/add:z:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/subSubPenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast:y:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/MaximumMaximumOenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub:z:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_1SubPenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_1/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum_1MaximumQenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_1:z:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_2Sub]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast_2CastUenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/truedivRealDivQenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/sub_2:z:0Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast_2:y:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/GreaterEqualGreaterEqualPenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast:y:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_1SelectV2Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/GreaterEqual:z:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum_1:z:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/CastCast[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/rangeRange\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/start:output:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/Cast:y:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast_3CastVenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1Range^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1/start:output:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/strided_slice:output:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1/delta:output:0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/EqualEqualYenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/range_1:output:0*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_2SelectV2Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Equal:z:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Maximum:z:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ReshapeReshapeRenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast_3:y:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_2:output:0*
T0*
_output_shapes
:?
Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/mulMulSenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/truediv:z:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Reshape:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/add_1AddV2[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims:output:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/mul:z:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/concatConcatV2[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/add_1:z:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/ExpandDims_1:output:0Yenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_3SelectV2Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Equal:z:0Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Shape_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SliceSliceWenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/concat:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/zeros_like:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/PowPowMenformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow/x:output:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace/Slice:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_3ReshapeFenformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_3/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_1AbsFenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs:y:0*
T0* 
_output_shapes
:
???
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_2LogOenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_2/x:output:0*
T0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/NegNegHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_2:y:0*
T0*
_output_shapes
: ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_1RealDivFenformer/trunk/transformer/transformer_block_2/mha/attention_2/Neg:y:0Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_3:output:0*
T0*"
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1StridedSliceHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_1:y:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stack:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stack_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_1MulLenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_1:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_1:output:0*
T0*$
_output_shapes
:???
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/ExpExpHenformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_1:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_2AbsUenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice:output:0*
T0* 
_output_shapes
:
???
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@?
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1RangeUenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/start:output:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/limit:output:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1/delta:output:0*

Tidx0*
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow_1PowOenformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow_1/x:output:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/range_1:output:0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/subSubHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Pow_1:z:0Menformer/trunk/transformer/transformer_block_2/mha/attention_2/sub/y:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_4ReshapeFenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_4/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_3AbsHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_2:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2StridedSliceHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_3:y:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stack:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stack_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/GreaterGreaterQenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_4:output:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_2:output:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_2/mha/attention_2/Cast_1CastJenformer/trunk/transformer/transformer_block_2/mha/attention_2/Greater:z:0*

DstT0*

SrcT0
*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_4AbsUenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice:output:0*
T0* 
_output_shapes
:
???
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @E?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/stopConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/CastCastVenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast_1CastRenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastArgsBroadcastArgsXenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape:output:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_1:output:0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastToBroadcastToXenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/start:output:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastTo_1BroadcastToWenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/stop:output:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims
ExpandDims^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastTo:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims_1
ExpandDims`enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/BroadcastTo_1:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_sliceStridedSliceZenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_3:output:0fenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stack:output:0henformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stack_1:output:0henformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/addAddV2`enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/add/y:output:0*
T0*
_output_shapes
: ?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2SelectV2eenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2/condition:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2/t:output:0Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/add:z:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/subSubRenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast:y:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/MaximumMaximumQenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub:z:0\enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_1SubRenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum_1MaximumSenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_1:z:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_2Sub_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast_2CastWenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/truedivRealDivSenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/sub_2:z:0Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/GreaterEqualGreaterEqualRenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast:y:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_1SelectV2Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/GreaterEqual:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum_1:z:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/CastCast]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/rangeRange^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/start:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/Cast:y:0^enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast_3CastXenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1Range`enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1/start:output:0`enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/strided_slice:output:0`enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1/delta:output:0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/EqualEqual[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2:output:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/range_1:output:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_2SelectV2Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Equal:z:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Maximum:z:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ReshapeReshapeTenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast_3:y:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/mulMulUenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/truediv:z:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Reshape:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/add_1AddV2]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/mul:z:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/concatConcatV2]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/add_1:z:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_3SelectV2Senformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Equal:z:0Renformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SliceSliceYenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/concat:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/zeros_like:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_5ReshapeXenformer/trunk/transformer/transformer_block_2/mha/attention_2/linspace_1/Slice:output:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_5/shape:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?D?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_2RealDivQenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_2/y:output:0*
T0*"
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2PowLenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_2:z:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2/y:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  J?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_3RealDivQenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_3/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_5AbsHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_4:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3StridedSliceHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Abs_5:y:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stack:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stack_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_1SubHenformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2:z:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_1/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/XlogyXlogyHenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_1:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_2MulLenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_3:z:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_2SubHenformer/trunk/transformer/transformer_block_2/mha/attention_2/Xlogy:z:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_2:z:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_2/mha/attention_2/LgammaLgammaHenformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_3LogLenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_3MulHenformer/trunk/transformer/transformer_block_2/mha/attention_2/pow_2:z:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Log_3:y:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_3SubIenformer/trunk/transformer/transformer_block_2/mha/attention_2/Lgamma:y:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_4SubHenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_2:z:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_3:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/Exp_1ExpHenformer/trunk/transformer/transformer_block_2/mha/attention_2/sub_4:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/addAddV2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Exp_1:y:0Menformer/trunk/transformer/transformer_block_2/mha/attention_2/add/y:output:0*
T0*$
_output_shapes
:???
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Benformer/trunk/transformer/transformer_block_2/mha/attention_2/MaxMaxFenformer/trunk/transformer/transformer_block_2/mha/attention_2/add:z:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/Max/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_4RealDivFenformer/trunk/transformer/transformer_block_2/mha/attention_2/add:z:0Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/Max:output:0*
T0*$
_output_shapes
:???
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Eenformer/trunk/transformer/transformer_block_2/mha/attention_2/concatConcatV2Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/Exp:y:0Ienformer/trunk/transformer/transformer_block_2/mha/attention_2/Cast_1:y:0Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/truediv_4:z:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/concat/axis:output:0*
N*
T0*$
_output_shapes
:???
Cenformer/trunk/transformer/transformer_block_2/mha/attention_2/SignSignUenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice:output:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4StridedSliceGenformer/trunk/transformer/transformer_block_2/mha/attention_2/Sign:y:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stack:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stack_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_4MulWenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_4:output:0Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat:output:0*
T0*$
_output_shapes
:???
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_1ConcatV2Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat:output:0Henformer/trunk/transformer/transformer_block_2/mha/attention_2/mul_4:z:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_1/axis:output:0*
N*
T0*$
_output_shapes
:???
Zenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?_     ?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/ReshapeReshapePenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_1:output:0cenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape/shape:output:0*
T0* 
_output_shapes
:
???
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOpReadVariableOpgenformer_trunk_transformer_transformer_block_2_mha_attention_2_r_k_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape:output:0fenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOp:value:0*
T0*!
_output_shapes
:????
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?_     ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape_1ReshapeYenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul:product:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape_1/shape:output:0*
T0*%
_output_shapes
:????
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?_     @   ?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_6/ReshapeReshape_enformer/trunk/transformer/transformer_block_2/mha/attention_2/batch_apply_3/Reshape_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:??@?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_3	TransposeYenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_6/Reshape:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_3/perm:output:0*
T0*(
_output_shapes
:??@?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_2_mha_attention_2_add_1_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1AddV2Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/mul:z:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Eenformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMulBatchMatMulV2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1:z:0Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_1:y:0*
T0*1
_output_shapes
:??????????`?`*
adj_y(?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_2_mha_attention_2_add_2_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2AddV2Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/mul:z:0[enformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul_1BatchMatMulV2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2:z:0Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_3:y:0*
T0*2
_output_shapes 
:??????????`??*
adj_y(?
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5StridedSlicePenformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stack:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stack_1:output:0_enformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????`*

begin_mask*
ellipsis_mask?
Ienformer/trunk/transformer/transformer_block_2/mha/attention_2/zeros_like	ZerosLikeWenformer/trunk/transformer/transformer_block_2/mha/attention_2/strided_slice_5:output:0*
T0*0
_output_shapes
:??????????`?
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_2ConcatV2Menformer/trunk/transformer/transformer_block_2/mha/attention_2/zeros_like:y:0Penformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul_1:output:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_2/axis:output:0*
N*
T0*2
_output_shapes 
:??????????`???
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    `   0  ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_7ReshapePenformer/trunk/transformer/transformer_block_2/mha/attention_2/concat_2:output:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_7/shape:output:0*
T0*2
_output_shapes 
:????????????`?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"               ?
Ienformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/SliceSliceQenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_7:output:0Senformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice/begin:output:0Renformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice/size:output:0*
Index0*
T0*2
_output_shapes 
:????????????`?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    0  ?_  ?
Henformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_8ReshapeMenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice:output:0Wenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_8/shape:output:0*
T0*2
_output_shapes 
:??????????`???
Lenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                ?
Kenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"???????????? 0  ?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1SliceQenformer/trunk/transformer/transformer_block_2/mha/attention_2/Reshape_8:output:0Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1/begin:output:0Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1/size:output:0*
Index0*
T0*1
_output_shapes
:??????????`?`?
Denformer/trunk/transformer/transformer_block_2/mha/attention_2/add_3AddV2Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul:output:0Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/Slice_1:output:0*
T0*1
_output_shapes
:??????????`?`?
Fenformer/trunk/transformer/transformer_block_2/mha/attention_2/SoftmaxSoftmaxHenformer/trunk/transformer/transformer_block_2/mha/attention_2/add_3:z:0*
T0*1
_output_shapes
:??????????`?`?
Genformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul_2BatchMatMulV2Penformer/trunk/transformer/transformer_block_2/mha/attention_2/Softmax:softmax:0Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_2:y:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_4	TransposePenformer/trunk/transformer/transformer_block_2/mha/attention_2/MatMul_2:output:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_4/perm:output:0*
T0*0
_output_shapes
:??????????`?
Nenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/ShapeShapeNenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_4:y:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/Shape:output:0eenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:??
Tenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concatConcatV2_enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/strided_slice:output:0aenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/ReshapeReshapeNenformer/trunk/transformer/transformer_block_2/mha/attention_2/transpose_4:y:0Xenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/concat:output:0*
T0*-
_output_shapes
:??????????`??
denformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOpReadVariableOpmenformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Uenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMulBatchMatMulV2Yenformer/trunk/transformer/transformer_block_2/mha/attention_2/reshape_9/Reshape:output:0lenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
aenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOpReadVariableOpjenformer_trunk_transformer_transformer_block_2_mha_attention_2_embedding_layer_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Renformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/AddAddV2^enformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul:output:0ienformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
;enformer/trunk/transformer/transformer_block_2/residual/addAddV2Aenformer/trunk/transformer/transformer_block_1/residual/add_1:z:0Venformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/meanMean?enformer/trunk/transformer/transformer_block_2/residual/add:z:0eenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/SquaredDifferenceSquaredDifference?enformer/trunk/transformer/transformer_block_2/residual/add:z:0[enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_2_mlp_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul_1Mul?enformer/trunk/transformer/transformer_block_2/residual/add:z:0Oenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_2_mlp_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Oenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOpReadVariableOpXenformer_trunk_transformer_transformer_block_2_mlp_linear_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
@enformer/trunk/transformer/transformer_block_2/mlp/linear/MatMulBatchMatMulV2Qenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/add_1:z:0Wenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOpReadVariableOpUenformer_trunk_transformer_transformer_block_2_mlp_linear_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
=enformer/trunk/transformer/transformer_block_2/mlp/linear/AddAddV2Ienformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul:output:0Tenformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
7enformer/trunk/transformer/transformer_block_2/mlp/ReluReluAenformer/trunk/transformer/transformer_block_2/mlp/linear/Add:z:0*
T0*-
_output_shapes
:??????????`??
Qenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOpReadVariableOpZenformer_trunk_transformer_transformer_block_2_mlp_linear_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Benformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1BatchMatMulV2Eenformer/trunk/transformer/transformer_block_2/mlp/Relu:activations:0Yenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOpReadVariableOpWenformer_trunk_transformer_transformer_block_2_mlp_linear_add_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
?enformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1AddV2Kenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1:output:0Venformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
=enformer/trunk/transformer/transformer_block_2/residual/add_1AddV2?enformer/trunk/transformer/transformer_block_2/residual/add:z:0Cenformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/meanMeanAenformer/trunk/transformer/transformer_block_2/residual/add_1:z:0eenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/SquaredDifferenceSquaredDifferenceAenformer/trunk/transformer/transformer_block_2/residual/add_1:z:0[enformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_3_mha_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul_1MulAenformer/trunk/transformer/transformer_block_2/residual/add_1:z:0Oenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_3/mha/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_3_mha_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/ShapeShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_sliceStridedSliceYenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stack:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stack_1:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/ProdProdaenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice:output:0Yenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1StridedSliceYenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stack:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stack_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concatConcatV2Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Prod:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/ReshapeReshapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_3_mha_attention_3_q_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMulMatMul[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Reshape:output:0denformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape_1ShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2StridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape_1:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stack:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stack_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape_2ShapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul:product:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3StridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Shape_2:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stack:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stack_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat_1ConcatV2cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_2:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/strided_slice_3:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul:product:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/ShapeShape]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Reshape_1:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_sliceStridedSliceUenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/Shape:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stack:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stack_1:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concatConcatV2]enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/strided_slice:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concat/values_1:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/ReshapeReshape]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply/Reshape_1:output:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose	TransposeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape/Reshape:output:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/ShapeShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/ProdProdcenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concatConcatV2Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Prod:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/ReshapeReshapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_3_mha_attention_3_k_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Reshape:output:0denformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape_1ShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape_2ShapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Shape_2:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/ShapeShape_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/Shape:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0     @   ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concatConcatV2_enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/strided_slice:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/ReshapeReshape_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_1/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/concat:output:0*
T0*0
_output_shapes
:??????????`@?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_1	TransposeYenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_1/Reshape:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_1/perm:output:0*
T0*0
_output_shapes
:??????????`@?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/ShapeShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
`enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_sliceStridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stack:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stack_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/ProdProdcenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1StridedSlice[enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concatConcatV2Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Prod:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_1:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/ReshapeReshapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat:output:0*
T0*(
_output_shapes
:???????????
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOpReadVariableOpeenformer_trunk_transformer_transformer_block_3_mha_attention_3_v_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Reshape:output:0denformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape_1ShapeQenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2StridedSlice]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape_1:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape_2ShapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul:product:0*
T0*
_output_shapes
:?
benformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3StridedSlice]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Shape_2:output:0kenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stack:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stack_1:output:0menformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat_1ConcatV2eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_2:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/strided_slice_3:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Reshape_1ReshapeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul:product:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/concat_1:output:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/ShapeShape_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Reshape_1:output:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/Shape:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB" 0        ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concatConcatV2_enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/strided_slice:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/ReshapeReshape_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_2/Reshape_1:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/concat:output:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_2	TransposeYenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_2/Reshape:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_2/perm:output:0*
T0*0
_output_shapes
:??????????`?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/mulMulLenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose:y:0Menformer/trunk/transformer/transformer_block_3/mha/attention_3/mul/y:output:0*
T0*0
_output_shapes
:??????????`@?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/range/startConst*
_output_shapes
: *
dtype0*
valueB
 * ????
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/rangeRangeSenformer/trunk/transformer/transformer_block_3/mha/attention_3/range/start:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/range/limit:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/range/delta:output:0*

Tidx0*
_output_shapes

:???
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_sliceStridedSliceMenformer/trunk/transformer/transformer_block_3/mha/attention_3/range:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stack:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stack_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*
new_axis_mask?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/AbsAbsUenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice:output:0*
T0* 
_output_shapes
:
???
Eenformer/trunk/transformer/transformer_block_3/mha/attention_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?`?
Cenformer/trunk/transformer/transformer_block_3/mha/attention_3/CastCastNenformer/trunk/transformer/transformer_block_3/mha/attention_3/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/LogLogGenformer/trunk/transformer/transformer_block_3/mha/attention_3/Cast:y:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_1LogOenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_1/x:output:0*
T0*
_output_shapes
: ?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/truedivRealDivFenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log:y:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_1:y:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @@?
Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/numConst*
_output_shapes
: *
dtype0*
value	B :?
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/CastCastTenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast_1CastPenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastArgsBroadcastArgsVenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_1:output:0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastToBroadcastToVenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/start:output:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastTo_1BroadcastToJenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv:z:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims
ExpandDims\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastTo:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims_1
ExpandDims^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/BroadcastTo_1:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_sliceStridedSliceXenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_3:output:0denformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stack:output:0fenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stack_1:output:0fenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/addAddV2^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice:output:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/add/y:output:0*
T0*
_output_shapes
: ?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2SelectV2cenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2/condition:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2/t:output:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/add:z:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/subSubPenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast:y:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/MaximumMaximumOenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub:z:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_1SubPenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_1/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum_1MaximumQenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_1:z:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_2Sub]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast_2CastUenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/truedivRealDivQenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/sub_2:z:0Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast_2:y:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/GreaterEqualGreaterEqualPenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast:y:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_1SelectV2Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/GreaterEqual:z:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum_1:z:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/CastCast[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/rangeRange\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/start:output:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/Cast:y:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast_3CastVenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1Range^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1/start:output:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/strided_slice:output:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1/delta:output:0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/EqualEqualYenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/range_1:output:0*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_2SelectV2Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Equal:z:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Maximum:z:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ReshapeReshapeRenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast_3:y:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_2:output:0*
T0*
_output_shapes
:?
Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/mulMulSenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/truediv:z:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Reshape:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/add_1AddV2[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims:output:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/mul:z:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/concatConcatV2[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/add_1:z:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/ExpandDims_1:output:0Yenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_3SelectV2Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Equal:z:0Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Cast:y:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Shape_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SliceSliceWenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/concat:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/zeros_like:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/PowPowMenformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow/x:output:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace/Slice:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_3ReshapeFenformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_3/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_1AbsFenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs:y:0*
T0* 
_output_shapes
:
???
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_2LogOenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_2/x:output:0*
T0*
_output_shapes
: ?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/NegNegHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_2:y:0*
T0*
_output_shapes
: ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_1RealDivFenformer/trunk/transformer/transformer_block_3/mha/attention_3/Neg:y:0Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_3:output:0*
T0*"
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1StridedSliceHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_1:y:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stack:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stack_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_1MulLenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_1:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_1:output:0*
T0*$
_output_shapes
:???
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/ExpExpHenformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_1:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_2AbsUenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice:output:0*
T0* 
_output_shapes
:
???
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@?
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1RangeUenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/start:output:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/limit:output:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1/delta:output:0*

Tidx0*
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow_1PowOenformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow_1/x:output:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/range_1:output:0*
T0*
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/subSubHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Pow_1:z:0Menformer/trunk/transformer/transformer_block_3/mha/attention_3/sub/y:output:0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_4ReshapeFenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_4/shape:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_3AbsHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_2:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2StridedSliceHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_3:y:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stack:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stack_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/GreaterGreaterQenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_4:output:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_2:output:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_3/mha/attention_3/Cast_1CastJenformer/trunk/transformer/transformer_block_3/mha/attention_3/Greater:z:0*

DstT0*

SrcT0
*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_4AbsUenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice:output:0*
T0* 
_output_shapes
:
???
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *  @E?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/stopConst*
_output_shapes
: *
dtype0*
valueB
 *  @F?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/CastCastVenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast_1CastRenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastArgsBroadcastArgsXenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape:output:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_1:output:0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastToBroadcastToXenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/start:output:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastTo_1BroadcastToWenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/stop:output:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: ?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims
ExpandDims^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastTo:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims_1
ExpandDims`enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/BroadcastTo_1:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:?
]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_sliceStridedSliceZenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_3:output:0fenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stack:output:0henformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stack_1:output:0henformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/addAddV2`enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/add/y:output:0*
T0*
_output_shapes
: ?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : ?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2SelectV2eenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2/condition:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2/t:output:0Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/add:z:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/subSubRenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast:y:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub/y:output:0*
T0*
_output_shapes
: ?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/MaximumMaximumQenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub:z:0\enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_1SubRenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum_1MaximumSenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_1:z:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_2Sub_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast_2CastWenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/truedivRealDivSenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/sub_2:z:0Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/GreaterEqualGreaterEqualRenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast:y:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_1SelectV2Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/GreaterEqual:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum_1:z:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: ?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/CastCast]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/rangeRange^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/start:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/Cast:y:0^enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast_3CastXenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : ?
Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1Range`enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1/start:output:0`enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/strided_slice:output:0`enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1/delta:output:0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/EqualEqual[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2:output:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/range_1:output:0*
T0*
_output_shapes
:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_2SelectV2Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Equal:z:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Maximum:z:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:?
Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ReshapeReshapeTenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast_3:y:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:?
Menformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/mulMulUenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/truediv:z:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Reshape:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/add_1AddV2]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims:output:0Qenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/mul:z:0*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/concatConcatV2]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/add_1:z:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/ExpandDims_1:output:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_3SelectV2Senformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Equal:z:0Renformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Cast:y:0Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SliceSliceYenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/concat:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/zeros_like:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_5ReshapeXenformer/trunk/transformer/transformer_block_3/mha/attention_3/linspace_1/Slice:output:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_5/shape:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?D?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_2RealDivQenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_2/y:output:0*
T0*"
_output_shapes
:?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2PowLenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_2:z:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2/y:output:0*
T0*"
_output_shapes
:?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  J?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_3RealDivQenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_5:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_3/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_5AbsHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_4:y:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3StridedSliceHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Abs_5:y:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stack:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stack_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_1SubHenformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2:z:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_1/y:output:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/XlogyXlogyHenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_1:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_2MulLenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_3:z:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_3:output:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_2SubHenformer/trunk/transformer/transformer_block_3/mha/attention_3/Xlogy:z:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_2:z:0*
T0*$
_output_shapes
:???
Eenformer/trunk/transformer/transformer_block_3/mha/attention_3/LgammaLgammaHenformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_3LogLenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_3MulHenformer/trunk/transformer/transformer_block_3/mha/attention_3/pow_2:z:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Log_3:y:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_3SubIenformer/trunk/transformer/transformer_block_3/mha/attention_3/Lgamma:y:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_3:z:0*
T0*"
_output_shapes
:?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_4SubHenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_2:z:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_3:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/Exp_1ExpHenformer/trunk/transformer/transformer_block_3/mha/attention_3/sub_4:z:0*
T0*$
_output_shapes
:???
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/addAddV2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Exp_1:y:0Menformer/trunk/transformer/transformer_block_3/mha/attention_3/add/y:output:0*
T0*$
_output_shapes
:???
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Benformer/trunk/transformer/transformer_block_3/mha/attention_3/MaxMaxFenformer/trunk/transformer/transformer_block_3/mha/attention_3/add:z:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/Max/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_4RealDivFenformer/trunk/transformer/transformer_block_3/mha/attention_3/add:z:0Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/Max:output:0*
T0*$
_output_shapes
:???
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Eenformer/trunk/transformer/transformer_block_3/mha/attention_3/concatConcatV2Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/Exp:y:0Ienformer/trunk/transformer/transformer_block_3/mha/attention_3/Cast_1:y:0Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/truediv_4:z:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/concat/axis:output:0*
N*
T0*$
_output_shapes
:???
Cenformer/trunk/transformer/transformer_block_3/mha/attention_3/SignSignUenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice:output:0*
T0* 
_output_shapes
:
???
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4StridedSliceGenformer/trunk/transformer/transformer_block_3/mha/attention_3/Sign:y:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stack:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stack_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4/stack_2:output:0*
Index0*
T0*$
_output_shapes
:??*
ellipsis_mask*
new_axis_mask?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_4MulWenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_4:output:0Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat:output:0*
T0*$
_output_shapes
:???
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_1ConcatV2Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat:output:0Henformer/trunk/transformer/transformer_block_3/mha/attention_3/mul_4:z:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_1/axis:output:0*
N*
T0*$
_output_shapes
:???
Zenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?_     ?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/ReshapeReshapePenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_1:output:0cenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape/shape:output:0*
T0* 
_output_shapes
:
???
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOpReadVariableOpgenformer_trunk_transformer_transformer_block_3_mha_attention_3_r_k_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMulMatMul]enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape:output:0fenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOp:value:0*
T0*!
_output_shapes
:????
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?_     ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape_1ReshapeYenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul:product:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape_1/shape:output:0*
T0*%
_output_shapes
:????
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   ?_     @   ?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_6/ReshapeReshape_enformer/trunk/transformer/transformer_block_3/mha/attention_3/batch_apply_3/Reshape_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_6/Reshape/shape:output:0*
T0*(
_output_shapes
:??@?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_3	TransposeYenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_6/Reshape:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_3/perm:output:0*
T0*(
_output_shapes
:??@?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_3_mha_attention_3_add_1_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1AddV2Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/mul:z:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Eenformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMulBatchMatMulV2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1:z:0Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_1:y:0*
T0*1
_output_shapes
:??????????`?`*
adj_y(?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOpReadVariableOp\enformer_trunk_transformer_transformer_block_3_mha_attention_3_add_2_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2AddV2Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/mul:z:0[enformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????`@?
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul_1BatchMatMulV2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2:z:0Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_3:y:0*
T0*2
_output_shapes 
:??????????`??*
adj_y(?
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5StridedSlicePenformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stack:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stack_1:output:0_enformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5/stack_2:output:0*
Index0*
T0*0
_output_shapes
:??????????`*

begin_mask*
ellipsis_mask?
Ienformer/trunk/transformer/transformer_block_3/mha/attention_3/zeros_like	ZerosLikeWenformer/trunk/transformer/transformer_block_3/mha/attention_3/strided_slice_5:output:0*
T0*0
_output_shapes
:??????????`?
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_2ConcatV2Menformer/trunk/transformer/transformer_block_3/mha/attention_3/zeros_like:y:0Penformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul_1:output:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_2/axis:output:0*
N*
T0*2
_output_shapes 
:??????????`???
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    `   0  ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_7ReshapePenformer/trunk/transformer/transformer_block_3/mha/attention_3/concat_2:output:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_7/shape:output:0*
T0*2
_output_shapes 
:????????????`?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice/beginConst*
_output_shapes
:*
dtype0*%
valueB"               ?
Ienformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice/sizeConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/SliceSliceQenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_7:output:0Senformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice/begin:output:0Renformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice/size:output:0*
Index0*
T0*2
_output_shapes 
:????????????`?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????    0  ?_  ?
Henformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_8ReshapeMenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice:output:0Wenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_8/shape:output:0*
T0*2
_output_shapes 
:??????????`???
Lenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1/beginConst*
_output_shapes
:*
dtype0*%
valueB"                ?
Kenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1/sizeConst*
_output_shapes
:*
dtype0*%
valueB"???????????? 0  ?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1SliceQenformer/trunk/transformer/transformer_block_3/mha/attention_3/Reshape_8:output:0Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1/begin:output:0Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1/size:output:0*
Index0*
T0*1
_output_shapes
:??????????`?`?
Denformer/trunk/transformer/transformer_block_3/mha/attention_3/add_3AddV2Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul:output:0Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/Slice_1:output:0*
T0*1
_output_shapes
:??????????`?`?
Fenformer/trunk/transformer/transformer_block_3/mha/attention_3/SoftmaxSoftmaxHenformer/trunk/transformer/transformer_block_3/mha/attention_3/add_3:z:0*
T0*1
_output_shapes
:??????????`?`?
Genformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul_2BatchMatMulV2Penformer/trunk/transformer/transformer_block_3/mha/attention_3/Softmax:softmax:0Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_2:y:0*
T0*0
_output_shapes
:??????????`?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_4	TransposePenformer/trunk/transformer/transformer_block_3/mha/attention_3/MatMul_2:output:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_4/perm:output:0*
T0*0
_output_shapes
:??????????`?
Nenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/ShapeShapeNenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_4:y:0*
T0*
_output_shapes
:?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Venformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_sliceStridedSliceWenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/Shape:output:0eenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stack:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stack_1:output:0genformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:??
Tenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Oenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concatConcatV2_enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/strided_slice:output:0aenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concat/values_1:output:0]enformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Penformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/ReshapeReshapeNenformer/trunk/transformer/transformer_block_3/mha/attention_3/transpose_4:y:0Xenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/concat:output:0*
T0*-
_output_shapes
:??????????`??
denformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOpReadVariableOpmenformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Uenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMulBatchMatMulV2Yenformer/trunk/transformer/transformer_block_3/mha/attention_3/reshape_9/Reshape:output:0lenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
aenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOpReadVariableOpjenformer_trunk_transformer_transformer_block_3_mha_attention_3_embedding_layer_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Renformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/AddAddV2^enformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul:output:0ienformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
;enformer/trunk/transformer/transformer_block_3/residual/addAddV2Aenformer/trunk/transformer/transformer_block_2/residual/add_1:z:0Venformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add:z:0*
T0*-
_output_shapes
:??????????`??
\enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/meanMean?enformer/trunk/transformer/transformer_block_3/residual/add:z:0eenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Renformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/StopGradientStopGradientSenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/mean:output:0*
T0*,
_output_shapes
:??????????`?
Wenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/SquaredDifferenceSquaredDifference?enformer/trunk/transformer/transformer_block_3/residual/add:z:0[enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/StopGradient:output:0*
T0*-
_output_shapes
:??????????`??
`enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Nenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/varianceMean[enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/SquaredDifference:z:0ienformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????`*
	keep_dims(?
Menformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Kenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/addAddV2Wenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/variance:output:0Venformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????`?
Menformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/RsqrtRsqrtOenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????`?
Zenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpcenformer_trunk_transformer_transformer_block_3_mlp_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mulMulQenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/Rsqrt:y:0benformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul_1Mul?enformer/trunk/transformer/transformer_block_3/residual/add:z:0Oenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul_2MulSenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/moments/mean:output:0Oenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul:z:0*
T0*-
_output_shapes
:??????????`??
Venformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOpReadVariableOp_enformer_trunk_transformer_transformer_block_3_mlp_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Kenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/subSub^enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOp:value:0Qenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul_2:z:0*
T0*-
_output_shapes
:??????????`??
Menformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/add_1AddV2Qenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul_1:z:0Oenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/sub:z:0*
T0*-
_output_shapes
:??????????`??
Oenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOpReadVariableOpXenformer_trunk_transformer_transformer_block_3_mlp_linear_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
@enformer/trunk/transformer/transformer_block_3/mlp/linear/MatMulBatchMatMulV2Qenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/add_1:z:0Wenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Lenformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOpReadVariableOpUenformer_trunk_transformer_transformer_block_3_mlp_linear_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
=enformer/trunk/transformer/transformer_block_3/mlp/linear/AddAddV2Ienformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul:output:0Tenformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
7enformer/trunk/transformer/transformer_block_3/mlp/ReluReluAenformer/trunk/transformer/transformer_block_3/mlp/linear/Add:z:0*
T0*-
_output_shapes
:??????????`??
Qenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOpReadVariableOpZenformer_trunk_transformer_transformer_block_3_mlp_linear_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Benformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1BatchMatMulV2Eenformer/trunk/transformer/transformer_block_3/mlp/Relu:activations:0Yenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
Nenformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOpReadVariableOpWenformer_trunk_transformer_transformer_block_3_mlp_linear_add_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
?enformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1AddV2Kenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1:output:0Venformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOp:value:0*
T0*-
_output_shapes
:??????????`??
=enformer/trunk/transformer/transformer_block_3/residual/add_1AddV2?enformer/trunk/transformer/transformer_block_3/residual/add:z:0Cenformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1:z:0*
T0*-
_output_shapes
:??????????`??
/enformer/trunk/target_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ?      ?
1enformer/trunk/target_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    ???    ?
1enformer/trunk/target_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ?
)enformer/trunk/target_input/strided_sliceStridedSliceAenformer/trunk/transformer/transformer_block_3/residual/add_1:z:08enformer/trunk/target_input/strided_slice/stack:output:0:enformer/trunk/target_input/strided_slice/stack_1:output:0:enformer/trunk/target_input/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:??????????*

begin_mask*
ellipsis_mask*
end_mask?
Xenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOpReadVariableOpaenformer_trunk_final_pointwise_conv_block_exponential_moving_average_read_readvariableop_resource*#
_output_shapes
:?*
dtype0?
Menformer/trunk/final_pointwise/conv_block/exponential_moving_average/IdentityIdentity`enformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
Zenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOpReadVariableOpcenformer_trunk_final_pointwise_conv_block_exponential_moving_average_read_1_readvariableop_resource*#
_output_shapes
:?*
dtype0?
Oenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Identity_1Identitybenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
Renformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Penformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/addAddV2Xenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Identity_1:output:0[enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add/y:output:0*
T0*#
_output_shapes
:??
Renformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/RsqrtRsqrtTenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add:z:0*
T0*#
_output_shapes
:??
_enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpReadVariableOphenformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Penformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mulMulVenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/Rsqrt:y:0genformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:??
Renformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul_1Mul2enformer/trunk/target_input/strided_slice:output:0Tenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*,
_output_shapes
:???????????
Renformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul_2MulVenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Identity:output:0Tenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul:z:0*
T0*#
_output_shapes
:??
[enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpReadVariableOpdenformer_trunk_final_pointwise_conv_block_cross_replica_batch_norm_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Penformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/subSubcenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp:value:0Venformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul_2:z:0*
T0*#
_output_shapes
:??
Renformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add_1AddV2Venformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul_1:z:0Tenformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????t
/enformer/trunk/final_pointwise/conv_block/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
-enformer/trunk/final_pointwise/conv_block/mulMul8enformer/trunk/final_pointwise/conv_block/mul/x:output:0Venformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*,
_output_shapes
:???????????
1enformer/trunk/final_pointwise/conv_block/SigmoidSigmoid1enformer/trunk/final_pointwise/conv_block/mul:z:0*
T0*,
_output_shapes
:???????????
/enformer/trunk/final_pointwise/conv_block/mul_1Mul5enformer/trunk/final_pointwise/conv_block/Sigmoid:y:0Venformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/add_1:z:0*
T0*,
_output_shapes
:???????????
Lenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Henformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims
ExpandDims3enformer/trunk/final_pointwise/conv_block/mul_1:z:0Uenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
Yenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpReadVariableOpbenformer_trunk_final_pointwise_conv_block_conv1_d_convolution_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Nenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Jenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1
ExpandDimsaenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp:value:0Wenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:???
=enformer/trunk/final_pointwise/conv_block/conv1_d/convolutionConv2DQenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims:output:0Senformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Eenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/SqueezeSqueezeFenformer/trunk/final_pointwise/conv_block/conv1_d/convolution:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
Henformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOpReadVariableOpQenformer_trunk_final_pointwise_conv_block_conv1_d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
9enformer/trunk/final_pointwise/conv_block/conv1_d/BiasAddBiasAddNenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/Squeeze:output:0Penformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
=enformer/trunk/final_pointwise/max_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
9enformer/trunk/final_pointwise/max_pooling1d_5/ExpandDims
ExpandDimsBenformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd:output:0Fenformer/trunk/final_pointwise/max_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:???????????
6enformer/trunk/final_pointwise/max_pooling1d_5/MaxPoolMaxPoolBenformer/trunk/final_pointwise/max_pooling1d_5/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
?
6enformer/trunk/final_pointwise/max_pooling1d_5/SqueezeSqueeze?enformer/trunk/final_pointwise/max_pooling1d_5/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
i
$enformer/trunk/final_pointwise/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *#????
"enformer/trunk/final_pointwise/mulMul-enformer/trunk/final_pointwise/mul/x:output:0?enformer/trunk/final_pointwise/max_pooling1d_5/Squeeze:output:0*
T0*,
_output_shapes
:???????????
&enformer/trunk/final_pointwise/SigmoidSigmoid&enformer/trunk/final_pointwise/mul:z:0*
T0*,
_output_shapes
:???????????
$enformer/trunk/final_pointwise/mul_1Mul*enformer/trunk/final_pointwise/Sigmoid:y:0?enformer/trunk/final_pointwise/max_pooling1d_5/Squeeze:output:0*
T0*,
_output_shapes
:??????????
'enformer/heads/head_yeast/flatten/ShapeShape(enformer/trunk/final_pointwise/mul_1:z:0*
T0*
_output_shapes
:
5enformer/heads/head_yeast/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7enformer/heads/head_yeast/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7enformer/heads/head_yeast/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/enformer/heads/head_yeast/flatten/strided_sliceStridedSlice0enformer/heads/head_yeast/flatten/Shape:output:0>enformer/heads/head_yeast/flatten/strided_slice/stack:output:0@enformer/heads/head_yeast/flatten/strided_slice/stack_1:output:0@enformer/heads/head_yeast/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
1enformer/heads/head_yeast/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:?o
-enformer/heads/head_yeast/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(enformer/heads/head_yeast/flatten/concatConcatV28enformer/heads/head_yeast/flatten/strided_slice:output:0:enformer/heads/head_yeast/flatten/concat/values_1:output:06enformer/heads/head_yeast/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:?
)enformer/heads/head_yeast/flatten/ReshapeReshape(enformer/trunk/final_pointwise/mul_1:z:01enformer/heads/head_yeast/flatten/concat:output:0*
T0*(
_output_shapes
:???????????
6enformer/heads/head_yeast/output/MatMul/ReadVariableOpReadVariableOp?enformer_heads_head_yeast_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
'enformer/heads/head_yeast/output/MatMulMatMul2enformer/heads/head_yeast/flatten/Reshape:output:0>enformer/heads/head_yeast/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
3enformer/heads/head_yeast/output/Add/ReadVariableOpReadVariableOp<enformer_heads_head_yeast_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
$enformer/heads/head_yeast/output/AddAddV21enformer/heads/head_yeast/output/MatMul:product:0;enformer/heads/head_yeast/output/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"enformer/heads/head_yeast/SoftplusSoftplus(enformer/heads/head_yeast/output/Add:z:0*
T0*'
_output_shapes
:?????????
IdentityIdentity0enformer/heads/head_yeast/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:??????????a
NoOpNoOp4^enformer/heads/head_yeast/output/Add/ReadVariableOp7^enformer/heads/head_yeast/output/MatMul/ReadVariableOpW^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOph^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpj^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpn^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpg^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOpi^enformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOpa^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpr^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpt^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpx^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpq^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOps^enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpW^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOph^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpj^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpn^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpg^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOpi^enformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOpa^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpr^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpt^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpx^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpq^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOps^enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpW^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOph^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpj^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpn^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpg^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOpi^enformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOpa^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpr^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpt^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpx^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpq^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOps^enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpW^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOph^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpj^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpn^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpg^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOpi^enformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOpa^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpr^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpt^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpx^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpq^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOps^enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpI^enformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOpZ^enformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp\^enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp`^enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpY^enformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOp[^enformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOp3^enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOpD^enformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOpH^enformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpY^enformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp[^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp_^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpX^enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpZ^enformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOpb^enformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOpe^enformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOp_^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOpM^enformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOpO^enformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOpP^enformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOpR^enformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOpb^enformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOpe^enformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOp_^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOpM^enformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOpO^enformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOpP^enformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOpR^enformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOpb^enformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOpe^enformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOp_^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOpM^enformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOpO^enformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOpP^enformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOpR^enformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOpT^enformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOpb^enformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOpe^enformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOp_^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOp]^enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOpW^enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOp[^enformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOpM^enformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOpO^enformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOpP^enformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOpR^enformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3enformer/heads/head_yeast/output/Add/ReadVariableOp3enformer/heads/head_yeast/output/Add/ReadVariableOp2p
6enformer/heads/head_yeast/output/MatMul/ReadVariableOp6enformer/heads/head_yeast/output/MatMul/ReadVariableOp2?
Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/BiasAdd/ReadVariableOp2?
genformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpgenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
ienformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpienformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
menformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
fenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read/ReadVariableOp2?
henformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOphenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp2?
qenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpqenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
senformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpsenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
wenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
penformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp2?
renformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOprenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/BiasAdd/ReadVariableOp2?
genformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpgenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
ienformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpienformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
menformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
fenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read/ReadVariableOp2?
henformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOphenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp2?
qenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpqenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
senformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpsenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
wenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
penformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp2?
renformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOprenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/BiasAdd/ReadVariableOp2?
genformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpgenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
ienformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpienformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
menformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
fenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read/ReadVariableOp2?
henformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOphenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp2?
qenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpqenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
senformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpsenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
wenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
penformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp2?
renformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOprenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOpVenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/BiasAdd/ReadVariableOp2?
genformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpgenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
ienformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpienformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
menformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpmenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
fenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOpfenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read/ReadVariableOp2?
henformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOphenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp2?
qenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpqenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
senformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpsenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
wenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOpwenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
penformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOppenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp2?
renformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOprenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
Henformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOpHenformer/trunk/final_pointwise/conv_block/conv1_d/BiasAdd/ReadVariableOp2?
Yenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpYenformer/trunk/final_pointwise/conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
[enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp[enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
_enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp_enformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
Xenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOpXenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read/ReadVariableOp2?
Zenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOpZenformer/trunk/final_pointwise/conv_block/exponential_moving_average/Read_1/ReadVariableOp2h
2enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOp2enformer/trunk/stem/conv1_d/BiasAdd/ReadVariableOp2?
Cenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOpCenformer/trunk/stem/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
Genformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOpGenformer/trunk/stem/pointwise_conv_block/conv1_d/BiasAdd/ReadVariableOp2?
Xenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOpXenformer/trunk/stem/pointwise_conv_block/conv1_d/convolution/ExpandDims_1/ReadVariableOp2?
Zenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOpZenformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/ReadVariableOp2?
^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp^enformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/batchnorm/mul/ReadVariableOp2?
Wenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOpWenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read/ReadVariableOp2?
Yenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOpYenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/Read_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOpSenformer/trunk/transformer/transformer_block_0/mha/attention_0/add_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOpSenformer/trunk/transformer/transformer_block_0/mha/attention_0/add_2/ReadVariableOp2?
aenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOpaenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/Add/ReadVariableOp2?
denformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOpdenformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/MatMul/ReadVariableOp2?
^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOp^enformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/MatMul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_0/mha/layer_norm/batchnorm/mul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/batchnorm/mul/ReadVariableOp2?
Lenformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOpLenformer/trunk/transformer/transformer_block_0/mlp/linear/Add/ReadVariableOp2?
Nenformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOpNenformer/trunk/transformer/transformer_block_0/mlp/linear/Add_1/ReadVariableOp2?
Oenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOpOenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul/ReadVariableOp2?
Qenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOpQenformer/trunk/transformer/transformer_block_0/mlp/linear/MatMul_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOpSenformer/trunk/transformer/transformer_block_1/mha/attention_1/add_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOpSenformer/trunk/transformer/transformer_block_1/mha/attention_1/add_2/ReadVariableOp2?
aenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOpaenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/Add/ReadVariableOp2?
denformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOpdenformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/MatMul/ReadVariableOp2?
^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOp^enformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/MatMul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_1/mha/layer_norm/batchnorm/mul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/batchnorm/mul/ReadVariableOp2?
Lenformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOpLenformer/trunk/transformer/transformer_block_1/mlp/linear/Add/ReadVariableOp2?
Nenformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOpNenformer/trunk/transformer/transformer_block_1/mlp/linear/Add_1/ReadVariableOp2?
Oenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOpOenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul/ReadVariableOp2?
Qenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOpQenformer/trunk/transformer/transformer_block_1/mlp/linear/MatMul_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOpSenformer/trunk/transformer/transformer_block_2/mha/attention_2/add_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOpSenformer/trunk/transformer/transformer_block_2/mha/attention_2/add_2/ReadVariableOp2?
aenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOpaenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/Add/ReadVariableOp2?
denformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOpdenformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/MatMul/ReadVariableOp2?
^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOp^enformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/MatMul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_2/mha/layer_norm/batchnorm/mul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/batchnorm/mul/ReadVariableOp2?
Lenformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOpLenformer/trunk/transformer/transformer_block_2/mlp/linear/Add/ReadVariableOp2?
Nenformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOpNenformer/trunk/transformer/transformer_block_2/mlp/linear/Add_1/ReadVariableOp2?
Oenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOpOenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul/ReadVariableOp2?
Qenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOpQenformer/trunk/transformer/transformer_block_2/mlp/linear/MatMul_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOpSenformer/trunk/transformer/transformer_block_3/mha/attention_3/add_1/ReadVariableOp2?
Senformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOpSenformer/trunk/transformer/transformer_block_3/mha/attention_3/add_2/ReadVariableOp2?
aenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOpaenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/Add/ReadVariableOp2?
denformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOpdenformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/MatMul/ReadVariableOp2?
^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOp^enformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/MatMul/ReadVariableOp2?
\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOp\enformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/MatMul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_3/mha/layer_norm/batchnorm/mul/ReadVariableOp2?
Venformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOpVenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/ReadVariableOp2?
Zenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOpZenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/batchnorm/mul/ReadVariableOp2?
Lenformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOpLenformer/trunk/transformer/transformer_block_3/mlp/linear/Add/ReadVariableOp2?
Nenformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOpNenformer/trunk/transformer/transformer_block_3/mlp/linear/Add_1/ReadVariableOp2?
Oenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOpOenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul/ReadVariableOp2?
Qenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOpQenformer/trunk/transformer/transformer_block_3/mlp/linear/MatMul_1/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameargs_0
?
M
1__inference_max_pooling1d_5_layer_call_fn_4768339

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768318v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768420

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_layer_call_fn_4768326

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768303v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling1d_2_layer_call_fn_4768425

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768374v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling1d_3_layer_call_fn_4768438

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768389v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768347

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768404

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768374

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
args_05
serving_default_args_0:0???????????9
yeast0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
\

_trunk

_heads
predict_on_batch

signatures"
_generic_user_object
+
_layers"
_generic_user_object
+
	yeast"
trackable_dict_wrapper
?
trace_02?
#__inference_predict_on_batch_791105?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *#? 
????????????ztrace_0
,
serving_default"
signature_map
C
	0

1
2
3
4"
trackable_list_wrapper
+
_layers"
_generic_user_object
?B?
#__inference_predict_on_batch_791105args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *#? 
????????????
?B?
%__inference_signature_wrapper_4768291args_0"?
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
 
+
_layers"
_generic_user_object
+
_layers"
_generic_user_object
+
_layers"
_generic_user_object
"
_generic_user_object
+
_layers"
_generic_user_object
.
0
1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
 0
!1
"2"
trackable_list_wrapper
"
_generic_user_object
,
#w
$b"
_generic_user_object
,
%w
&b"
_generic_user_object
+
'_module"
_generic_user_object
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
+
._layers"
_generic_user_object
+
/_layers"
_generic_user_object
+
0_layers"
_generic_user_object
+
1_layers"
_generic_user_object
+
2_layers"
_generic_user_object
+
3_layers"
_generic_user_object
+
4_layers"
_generic_user_object
+
5_layers"
_generic_user_object
+
6_layers"
_generic_user_object
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
5:3	?2"enformer/heads/head_yeast/output/w
0:.2"enformer/heads/head_yeast/output/b
3:102enformer/trunk/stem/conv1_d/w
+:)02enformer/trunk/stem/conv1_d/b
+
=_layers"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
Ctrace_02?
/__inference_max_pooling1d_layer_call_fn_4768326?
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
 zCtrace_0
?
Dtrace_02?
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768334?
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
 zDtrace_0
5
E0
F1
G2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
K0
L1
M2"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
Y0
Z2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?
`trace_02?
1__inference_max_pooling1d_5_layer_call_fn_4768339?
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
 z`trace_0
?
atrace_02?
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768347?
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
 zatrace_0
.
b0
c2"
trackable_list_wrapper
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
?B?
/__inference_max_pooling1d_layer_call_fn_4768326inputs"?
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
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768334inputs"?
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
+
d_layers"
_generic_user_object
+
e_module"
_generic_user_object
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
+
l_layers"
_generic_user_object
+
m_module"
_generic_user_object
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
+
t_layers"
_generic_user_object
+
u_module"
_generic_user_object
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
+
|_layers"
_generic_user_object
+
}_module"
_generic_user_object
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
,
?_module"
_generic_user_object
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
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
?B?
1__inference_max_pooling1d_5_layer_call_fn_4768339inputs"?
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
?B?
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768347inputs"?
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
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
0
?0
?2"
trackable_list_wrapper
,
?_layers"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling1d_1_layer_call_fn_4768412?
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
 z?trace_0
?
?trace_02?
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768420?
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
 z?trace_0
0
?0
?2"
trackable_list_wrapper
,
?_layers"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling1d_2_layer_call_fn_4768425?
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
 z?trace_0
?
?trace_02?
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768433?
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
 z?trace_0
0
?0
?2"
trackable_list_wrapper
,
?_layers"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling1d_3_layer_call_fn_4768438?
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
 z?trace_0
?
?trace_02?
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768446?
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
 z?trace_0
0
?0
?2"
trackable_list_wrapper
,
?_layers"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling1d_4_layer_call_fn_4768451?
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
 z?trace_0
?
?trace_02?
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768459?
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
 z?trace_0
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
,
?_layers"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
W:U?2Henformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/scale
X:V?2Ienformer/trunk/final_pointwise/conv_block/cross_replica_batch_norm/offset
K:I??23enformer/trunk/final_pointwise/conv_block/conv1_d/w
B:@?23enformer/trunk/final_pointwise/conv_block/conv1_d/b
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
U:S02Genformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/scale
V:T02Henformer/trunk/stem/pointwise_conv_block/cross_replica_batch_norm/offset
H:F0022enformer/trunk/stem/pointwise_conv_block/conv1_d/w
@:>022enformer/trunk/stem/pointwise_conv_block/conv1_d/b
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
0
?0
?2"
trackable_list_wrapper
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
?B?
1__inference_max_pooling1d_1_layer_call_fn_4768412inputs"?
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
?B?
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768420inputs"?
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
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
0
?0
?2"
trackable_list_wrapper
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
?B?
1__inference_max_pooling1d_2_layer_call_fn_4768425inputs"?
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
?B?
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768433inputs"?
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
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
0
?0
?2"
trackable_list_wrapper
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
?B?
1__inference_max_pooling1d_3_layer_call_fn_4768438inputs"?
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
?B?
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768446inputs"?
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
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
0
?0
?2"
trackable_list_wrapper
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
?B?
1__inference_max_pooling1d_4_layer_call_fn_4768451inputs"?
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
?B?
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768459inputs"?
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
8
?0
?1
?2"
trackable_list_wrapper
H
?0
?1
?2
?4
?5"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
H
?0
?1
?2
?4
?5"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
H
?0
?1
?2
?4
?5"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
H
?0
?1
?2
?4
?5"
trackable_list_wrapper
T:R	 2Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter
`:^?2Kenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden
a:_?2Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average
T:R	 2Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/counter
`:^?2Kenformer/trunk/final_pointwise/conv_block/exponential_moving_average/hidden
a:_?2Lenformer/trunk/final_pointwise/conv_block/exponential_moving_average/average
S:Q	 2Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter
^:\02Jenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden
_:]02Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average
S:Q	 2Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/counter
^:\02Jenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/hidden
_:]02Kenformer/trunk/stem/pointwise_conv_block/exponential_moving_average/average
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
d:b02Venformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/scale
e:c02Wenformer/trunk/conv_tower/conv_tower_block_0/conv_block/cross_replica_batch_norm/offset
W:U0@2Aenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/w
O:M@2Aenformer/trunk/conv_tower/conv_tower_block_0/conv_block/conv1_d/b
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
d:b@2Venformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/scale
e:c@2Wenformer/trunk/conv_tower/conv_tower_block_1/conv_block/cross_replica_batch_norm/offset
W:U@@2Aenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/w
O:M@2Aenformer/trunk/conv_tower/conv_tower_block_1/conv_block/conv1_d/b
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
d:b@2Venformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/scale
e:c@2Wenformer/trunk/conv_tower/conv_tower_block_2/conv_block/cross_replica_batch_norm/offset
X:V@?2Aenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/w
P:N?2Aenformer/trunk/conv_tower/conv_tower_block_2/conv_block/conv1_d/b
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
e:c?2Venformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/scale
f:d?2Wenformer/trunk/conv_tower/conv_tower_block_3/conv_block/cross_replica_batch_norm/offset
Y:W??2Aenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/w
P:N?2Aenformer/trunk/conv_tower/conv_tower_block_3/conv_block/conv1_d/b
_
?moving_mean
?moving_variance

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
7

?scale
?offset"
_generic_user_object
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
?
!?_relative_position_functions
?_q_layer
?_k_layer
?_v_layer
?_embedding_layer
?
_r_k_layer
?	_r_w_bias
?	_r_r_bias"
_generic_user_object
"
_generic_user_object
7

?scale
?offset"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
.
?w
?b"
_generic_user_object
"
_generic_user_object
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter
m:k02Yenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden
n:l02Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/counter
m:k02Yenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/hidden
n:l02Zenformer/trunk/conv_tower/conv_tower_block_0/conv_block/exponential_moving_average/average
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
n:l@2`enformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/scale
o:m@2aenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/cross_replica_batch_norm/offset
a:_@@2Kenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/w
Y:W@2Kenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/conv1_d/b
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter
m:k@2Yenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden
n:l@2Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/counter
m:k@2Yenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/hidden
n:l@2Zenformer/trunk/conv_tower/conv_tower_block_1/conv_block/exponential_moving_average/average
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
n:l@2`enformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/scale
o:m@2aenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/cross_replica_batch_norm/offset
a:_@@2Kenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/w
Y:W@2Kenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/conv1_d/b
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter
m:k@2Yenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden
n:l@2Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/counter
m:k@2Yenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/hidden
n:l@2Zenformer/trunk/conv_tower/conv_tower_block_2/conv_block/exponential_moving_average/average
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
o:m?2`enformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/scale
p:n?2aenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/cross_replica_batch_norm/offset
c:a??2Kenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/w
Z:X?2Kenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/conv1_d/b
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter
n:l?2Yenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden
o:m?2Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average
b:`	 2Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/counter
n:l?2Yenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/hidden
o:m?2Zenformer/trunk/conv_tower/conv_tower_block_3/conv_block/exponential_moving_average/average
I
?_counter
?_hidden
?average"
_generic_user_object
I
?_counter
?_hidden
?average"
_generic_user_object
o:m?2`enformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/scale
p:n?2aenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/cross_replica_batch_norm/offset
c:a??2Kenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/w
Z:X?2Kenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/conv1_d/b
R:P?2Cenformer/trunk/transformer/transformer_block_0/mha/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_0/mha/layer_norm/offset
 "
trackable_list_wrapper
&
?w"
_generic_user_object
&
?w"
_generic_user_object
&
?w"
_generic_user_object
.
?w
?b"
_generic_user_object
&
?w"
_generic_user_object
a:_@2Genformer/trunk/transformer/transformer_block_0/mha/attention_0/r_w_bias
a:_@2Genformer/trunk/transformer/transformer_block_0/mha/attention_0/r_r_bias
R:P?2Cenformer/trunk/transformer/transformer_block_0/mlp/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_0/mlp/layer_norm/offset
O:M
??2;enformer/trunk/transformer/transformer_block_0/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_0/mlp/linear/b
O:M
??2;enformer/trunk/transformer/transformer_block_0/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_0/mlp/linear/b
R:P?2Cenformer/trunk/transformer/transformer_block_1/mha/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_1/mha/layer_norm/offset
 "
trackable_list_wrapper
&
?w"
_generic_user_object
&
?w"
_generic_user_object
&
?w"
_generic_user_object
.
?w
?b"
_generic_user_object
&
?w"
_generic_user_object
a:_@2Genformer/trunk/transformer/transformer_block_1/mha/attention_1/r_w_bias
a:_@2Genformer/trunk/transformer/transformer_block_1/mha/attention_1/r_r_bias
R:P?2Cenformer/trunk/transformer/transformer_block_1/mlp/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_1/mlp/layer_norm/offset
O:M
??2;enformer/trunk/transformer/transformer_block_1/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_1/mlp/linear/b
O:M
??2;enformer/trunk/transformer/transformer_block_1/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_1/mlp/linear/b
R:P?2Cenformer/trunk/transformer/transformer_block_2/mha/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_2/mha/layer_norm/offset
 "
trackable_list_wrapper
&
?w"
_generic_user_object
&
?w"
_generic_user_object
&
?w"
_generic_user_object
.
?w
?b"
_generic_user_object
&
?w"
_generic_user_object
a:_@2Genformer/trunk/transformer/transformer_block_2/mha/attention_2/r_w_bias
a:_@2Genformer/trunk/transformer/transformer_block_2/mha/attention_2/r_r_bias
R:P?2Cenformer/trunk/transformer/transformer_block_2/mlp/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_2/mlp/layer_norm/offset
O:M
??2;enformer/trunk/transformer/transformer_block_2/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_2/mlp/linear/b
O:M
??2;enformer/trunk/transformer/transformer_block_2/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_2/mlp/linear/b
R:P?2Cenformer/trunk/transformer/transformer_block_3/mha/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_3/mha/layer_norm/offset
 "
trackable_list_wrapper
&
?w"
_generic_user_object
&
?w"
_generic_user_object
&
?w"
_generic_user_object
.
?w
?b"
_generic_user_object
&
?w"
_generic_user_object
a:_@2Genformer/trunk/transformer/transformer_block_3/mha/attention_3/r_w_bias
a:_@2Genformer/trunk/transformer/transformer_block_3/mha/attention_3/r_r_bias
R:P?2Cenformer/trunk/transformer/transformer_block_3/mlp/layer_norm/scale
S:Q?2Denformer/trunk/transformer/transformer_block_3/mlp/layer_norm/offset
O:M
??2;enformer/trunk/transformer/transformer_block_3/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_3/mlp/linear/b
O:M
??2;enformer/trunk/transformer/transformer_block_3/mlp/linear/w
J:H?2;enformer/trunk/transformer/transformer_block_3/mlp/linear/b
l:j	 2denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter
w:u@2cenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden
x:v@2denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/counter
w:u@2cenformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/hidden
x:v@2denformer/trunk/conv_tower/conv_tower_block_0/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter
w:u@2cenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden
x:v@2denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/counter
w:u@2cenformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/hidden
x:v@2denformer/trunk/conv_tower/conv_tower_block_1/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter
x:v?2cenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden
y:w?2denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/counter
x:v?2cenformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/hidden
y:w?2denformer/trunk/conv_tower/conv_tower_block_2/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter
x:v?2cenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden
y:w?2denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average
l:j	 2denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/counter
x:v?2cenformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/hidden
y:w?2denformer/trunk/conv_tower/conv_tower_block_3/pointwise_conv_block/exponential_moving_average/average
\:Z
??2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/q_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_0/mha/attention_0/v_layer/w
d:b
??2Penformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/w
_:]?2Penformer/trunk/transformer/transformer_block_0/mha/attention_0/embedding_layer/b
]:[	?2Jenformer/trunk/transformer/transformer_block_0/mha/attention_0/r_k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/q_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_1/mha/attention_1/v_layer/w
d:b
??2Penformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/w
_:]?2Penformer/trunk/transformer/transformer_block_1/mha/attention_1/embedding_layer/b
]:[	?2Jenformer/trunk/transformer/transformer_block_1/mha/attention_1/r_k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/q_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_2/mha/attention_2/v_layer/w
d:b
??2Penformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/w
_:]?2Penformer/trunk/transformer/transformer_block_2/mha/attention_2/embedding_layer/b
]:[	?2Jenformer/trunk/transformer/transformer_block_2/mha/attention_2/r_k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/q_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/k_layer/w
\:Z
??2Henformer/trunk/transformer/transformer_block_3/mha/attention_3/v_layer/w
d:b
??2Penformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/w
_:]?2Penformer/trunk/transformer/transformer_block_3/mha/attention_3/embedding_layer/b
]:[	?2Jenformer/trunk/transformer/transformer_block_3/mha/attention_3/r_k_layer/w?
L__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_4768420?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_1_layer_call_fn_4768412wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_4768433?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_2_layer_call_fn_4768425wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_4768446?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_3_layer_call_fn_4768438wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_4_layer_call_and_return_conditional_losses_4768459?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_4_layer_call_fn_4768451wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_5_layer_call_and_return_conditional_losses_4768347?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_5_layer_call_fn_4768339wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
J__inference_max_pooling1d_layer_call_and_return_conditional_losses_4768334?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
/__inference_max_pooling1d_layer_call_fn_4768326wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
#__inference_predict_on_batch_791105??%&????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????#$5?2
+?(
&?#
args_0???????????
? "-?*
(
yeast?
yeast??????????
%__inference_signature_wrapper_4768291??%&????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????#$??<
? 
5?2
0
args_0&?#
args_0???????????"-?*
(
yeast?
yeast?????????