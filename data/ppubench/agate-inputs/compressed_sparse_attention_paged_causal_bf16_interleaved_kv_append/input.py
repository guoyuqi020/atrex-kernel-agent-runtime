from __future__ import annotations
import math
import torch
CACHE_BLOCKS=1282
BLOCK_SIZE=1792
HEAD_DIM=256
QUERY_HEADS=12
INDEX_WIDTH=2051
MAX_QUERY_TOKENS=8192
TABLE_WIDTH=586
_DRAFT_QUANTILES={'query':(-83.0,-10.25,-0.065673828125,7.5625,21.125),'key':(-87.0,-4.409375190734863,-0.05419921875,3.015625,4.875),'value':(-4.75,-2.90625,0.0751953125,2.921875,4.59375)}
def _sample(shape,name,profile,rng):
  if profile=='draft_decode':
    quant=_DRAFT_QUANTILES[name]
    tail_exp={'query':0.5,'key':4.0,'value':0.05}[name]
    inner_exp={'query':8.0,'key':8.0,'value':2.0}[name]
    u=torch.rand(shape,generator=rng,dtype=torch.float32,device='cpu')
    values=torch.empty_like(u)
    cutoffs=(0.0,0.01,0.5,0.99,1.0)
    for i in range(4):
      lo,hi=(cutoffs[i],cutoffs[i+1])
      mask=(u>=lo)&(u<hi)
      f=(u[mask]-lo)/(hi-lo)
      if i==0:
        f=f.pow(tail_exp)
      elif i==1:
        f=1.0-(1.0-f).pow(inner_exp)
      elif i==2:
        f=f.pow(inner_exp)
      elif i==3:
        f=f.pow(4.0)
      values[mask]=quant[i]+f*(quant[i+1]-quant[i])
    return values
  if name=='query':
    mean,std=(0.049,1.178)if profile=='main_prefill' else(0.014,1.214)
  elif name=='key':
    mean,std=(-0.018,1.348)if profile=='main_prefill' else(-0.057,1.35)
  else:
    mean,std=(0.016,0.48)if profile=='main_prefill' else(0.028,0.489)
  return torch.randn(shape,generator=rng,dtype=torch.float32,device='cpu')*std+mean
def _qlens(query_tokens,requests,profile):
  if profile=='main_prefill':
    if requests==1:
      return[query_tokens]
    if query_tokens!=MAX_QUERY_TOKENS or not 11<=requests<=16:
      raise ValueError('prefill geometry differs from selected C16 Shapes')
    return[0]*(requests-1)+[query_tokens]
  if profile=='main_decode':
    if query_tokens!=4*requests:
      raise ValueError('main decode must have four query tokens per request')
    return[4]*requests
  if profile=='draft_decode':
    if not requests<=query_tokens<=4*requests:
      raise ValueError('draft decode query/request geometry differs')
    result=[1]*requests
    left=query_tokens-requests
    for i in range(requests):
      take=min(3,left)
      result[i]+=take
      left-=take
    if left:
      raise ValueError('draft decode query tokens are not conserved')
    return result
  raise ValueError('CSA profile is not source-calibrated')
def _slens(qlens,profile):
  obs=[240,509,712,779,933,654,933,407,437,370,717,796,323,715,702,363]if profile=='draft_decode' else[240,509,713,780,934,656,935,407,439,371,717,796,324,718,705,366]
  indices=torch.linspace(0,15,len(qlens)).round().long().tolist()
  lengths=[obs[i]for i in indices]
  if profile=='main_prefill':
    requests=len(qlens)
    if requests==1:
      return qlens.copy()
    lengths[-1]=min(65532,qlens[-1]*(17-requests))
  return[max(length,queries)for length,queries in zip(lengths,qlens)]
def _indices(pos):
  complete=(pos+1)//4*4
  n_base=min(complete,2048)
  if complete<=2048:
    blocks=torch.arange(n_base//4,dtype=torch.int32,device='cpu')
  else:
    recent_frac=max(0.1,min(0.25,0.264-1.3e-05*pos))
    n_blocks=n_base//4
    n_recent=min(128,round(n_blocks*recent_frac))
    n_old=complete//4-128
    n_older=min(n_blocks-n_recent,n_old)
    n_recent=n_blocks-n_older
    older=torch.linspace(0,n_old-1,n_older).round().to(torch.int32)
    recent=torch.linspace(complete//4-128,complete//4-1,n_recent).round().to(torch.int32)
    blocks=torch.cat((older,recent))
  rng=torch.Generator(device='cpu').manual_seed(2017+pos*37)
  blocks=blocks[torch.randperm(len(blocks),generator=rng,device='cpu')]
  base=(blocks[:,None]*4+torch.arange(4,dtype=torch.int32,device='cpu')).reshape(-1)
  tail=torch.arange(complete,pos+1,dtype=torch.int32,device='cpu')
  return torch.cat((base,tail))
def _make_inputs(query_tokens:int,requests:int,profile:str,dtype:str='bfloat16',device:str='cuda',seed:int|None=None)->dict[str,object]:
  if type(query_tokens)is not int or type(requests)is not int:
    raise TypeError('CSA query and request dimensions must be integers')
  if not 1<=query_tokens<=MAX_QUERY_TOKENS or not 1<=requests<=16:
    raise ValueError('CSA geometry is outside the observed C16 ABI')
  if dtype!='bfloat16':
    raise ValueError('CSA dtype differs from the observed C16 ABI')
  qlens=_qlens(query_tokens,requests,profile)
  slens=_slens(qlens,profile)
  rng=torch.Generator(device='cpu').manual_seed(torch.initial_seed()%2**63 if seed is None else seed)
  query=_sample((query_tokens,QUERY_HEADS,HEAD_DIM),'query',profile,rng)
  key=_sample((query_tokens,1,HEAD_DIM),'key',profile,rng)
  values=_sample((query_tokens,1,HEAD_DIM),'value',profile,rng)
  query=query.to(dtype=torch.bfloat16,device=device)
  key=key.to(dtype=torch.bfloat16,device=device)
  v_buf=torch.zeros(query_tokens*6656,dtype=torch.bfloat16,device=device)
  value=torch.as_strided(v_buf,size=(query_tokens,1,HEAD_DIM),stride=(6656,HEAD_DIM,1),storage_offset=6400)
  value.copy_(values.to(dtype=torch.bfloat16,device=device))
  out_std=0.84 if profile=='draft_decode' else 0.465 if profile=='main_prefill' else 0.598
  output=(torch.randn((query_tokens,QUERY_HEADS,HEAD_DIM),generator=rng,device='cpu',dtype=torch.float32)*out_std).to(dtype=torch.bfloat16,device=device)
  cache=torch.zeros((CACHE_BLOCKS,2,BLOCK_SIZE,1,HEAD_DIM),dtype=torch.bfloat16,device=device)
  kv_cache=cache.permute(1,0,2,3,4)
  block_table=torch.zeros((requests,TABLE_WIDTH),dtype=torch.int32,device=device)
  pages_by_req=[]
  next_pg=180
  for req,length in enumerate(slens):
    count=math.ceil(length/BLOCK_SIZE)
    if next_pg+count>CACHE_BLOCKS:
      raise ValueError('source-valid CSA page allocation exceeded the cache')
    pages=list(range(next_pg,next_pg+count))
    pages_by_req.append(pages)
    block_table[req,:count]=torch.tensor(pages,dtype=torch.int32,device=device)
    next_pg+=count
  reqs=[]
  pos_host=[]
  slots=[]
  for req,(length,queries)in enumerate(zip(slens,qlens)):
    prior=length-queries
    if prior:
      positions=torch.arange(prior,dtype=torch.long,device=device)
      pages=block_table[req,positions//BLOCK_SIZE].long()
      offsets=positions.remainder(BLOCK_SIZE)
      cache_profile='draft_decode' if profile=='draft_decode' else 'main_decode'
      prior_k=_sample((prior,1,HEAD_DIM),'key',cache_profile,rng)
      prior_v=_sample((prior,1,HEAD_DIM),'value',cache_profile,rng)
      kv_cache[0,pages,offsets,0]=prior_k[:,0].to(dtype=torch.bfloat16,device=device)
      kv_cache[1,pages,offsets,0]=prior_v[:,0].to(dtype=torch.bfloat16,device=device)
    for pos in range(prior,length):
      reqs.append(req)
      pos_host.append(pos)
      page=pages_by_req[req][pos//BLOCK_SIZE]
      slots.append(page*BLOCK_SIZE+pos%BLOCK_SIZE)
  if len(slots)!=query_tokens:
    raise ValueError('CSA slot mapping does not conserve query tokens')
  slot_buf=torch.zeros(MAX_QUERY_TOKENS,dtype=torch.int64,device=device)
  slot_buf[:query_tokens]=torch.tensor(slots,dtype=torch.int64,device=device)
  slot_mapping=slot_buf[:query_tokens]
  req_buf=torch.zeros(MAX_QUERY_TOKENS,dtype=torch.int32,device=device)
  req_buf[:query_tokens]=torch.tensor(reqs,dtype=torch.int32,device=device)
  req_ids=req_buf[:query_tokens]
  seq_lens=torch.tensor(slens,dtype=torch.int32,device=device)
  topk_cpu=torch.full((MAX_QUERY_TOKENS,INDEX_WIDTH),-1,dtype=torch.int32,device='cpu')
  for row,pos in enumerate(pos_host):
    chosen=_indices(pos)
    if chosen.numel()>INDEX_WIDTH or(chosen.numel()and int(chosen[-1])>=slens[reqs[row]]):
      raise ValueError('CSA selected token controls are out of range')
    topk_cpu[row,:chosen.numel()]=chosen
  topk_indices=topk_cpu.to(device=device)
  return{'query':query,'key':key,'value':value,'kv_cache':kv_cache,'output':output,'topk_indices_buffer':topk_indices,'metadata_present':True,'num_actual_tokens':query_tokens,'skip_write_kvcache':False,'slot_mapping':slot_mapping,'block_table':block_table,'req_id_per_token':req_ids,'seq_lens':seq_lens,'num_reqs':requests,'output_scale':None,'output_block_scale':None}
