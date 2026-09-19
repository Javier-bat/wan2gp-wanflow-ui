(() => {
  const graph = window.W2GP_INITIAL_GRAPH || {nodes: [], edges: []};
  const catalog = window.W2GP_NODE_CATALOG || {nodes: [], models: [], processes: []};
  graph.groups = Array.isArray(graph.groups) ? graph.groups : [];
  let selected = null;
  let selectedNodes = [];
  let selectedGroup = null;
  let pendingDeleteBlockId = null;
  let selectedEdge = null;
  let pending = null;
  let scale = 1;
  let pan = {x: 0, y: 0};
  let dragging = null;
  let panning = null;
  let runtimePreviews = [];
  let executionState = {nodes: {}};
  const $ = id => document.getElementById(id);
  const nodeById = id => graph.nodes.find(node => node.id === id);
  const groupById = id => graph.groups.find(group => group.id === id);
  const groupForNode = id => graph.groups.find(group => (group.node_ids||[]).includes(id));
  function selectNode(id, additive=false){selectedGroup=null;if(additive){selectedNodes=selectedNodes.includes(id)?selectedNodes.filter(value=>value!==id):[...selectedNodes,id];}else{selectedNodes=[id];}selected=selectedNodes[0]||null;}
  function selectGroup(id){selectedGroup=id;selected=null;selectedNodes=[];}
  const modelById = id => catalog.models.find(model => model.model_type === id);
  const generationTypes = ['generate_image','edit_image','inpaint_image','generate_video','edit_video','inpaint_video','generate_audio'];
  const typeClass = type => String(type || 'MEDIA').replace(/[^A-Za-z0-9_]/g, '');
  const portBase = type => String(type || 'MEDIA').replace(/\[\]$/, '');

  function modelSupportsNode(model,nodeType){
    const caps=model?.capabilities||{}, outputs=(caps.main_output||caps.outputs||[]).map(value=>String(value).toUpperCase());
    const expected=nodeType.includes('image')?'IMAGE':nodeType.includes('video')?'VIDEO':nodeType.includes('audio')?'AUDIO':'';
    if(expected && !outputs.includes(expected)) return false;
    if(nodeType.startsWith('inpaint_')) return Boolean(caps.inpainting);
    if(nodeType==='edit_image') return Boolean(caps.image_to_image||caps.reference_images||caps.control_image);
    if(nodeType==='edit_video') return Boolean(caps.video_to_video||caps.video_continuation||caps.reference_videos||caps.control_video);
    return true;
  }

  function modelChoicesFor(node){return catalog.models.filter(model=>modelSupportsNode(model,node.type));}
  function modelLabel(model){
    const outputs=(model?.capabilities?.main_output||[]).join('+');
    return outputs ? `${model.name||model.model_type} [${outputs}]` : (model.name||model.model_type);
  }
  function nativeSettingId(key){return `editNative_${String(key).replace(/[^A-Za-z0-9_]/g,'_')}`;}
  function nativeSettingValue(params,setting){
    const settings=params.settings||{};
    const source=setting.custom?(settings.custom_settings||{}):settings;
    return Object.prototype.hasOwnProperty.call(source,setting.key)?source[setting.key]:'';
  }
  function nativeSettingHtml(setting,params){
    const id=nativeSettingId(setting.key), value=nativeSettingValue(params,setting), label=setting.label||setting.key;
    const disabled=setting.disabled?' disabled':'';
    if(Array.isArray(setting.choices)&&setting.choices.length){
      const choices=setting.choices.some(option=>String(option.value)==='')?setting.choices:[{label:'Wan2GP default',value:''},...setting.choices];
      return `<label>${escapeHtml(label)}</label><select id="${id}"${disabled}>${choices.map(option=>`<option value="${escapeAttr(option.value??'')}" ${String(option.value??'')===String(value??'')?'selected':''}>${escapeHtml(option.label??option.value??'')}</option>`).join('')}</select>`;
    }
    const type=setting.type==='integer'||setting.type==='number'?'number':'text';
    const attrs=type==='number'?` min="${escapeAttr(setting.min??'')}" max="${escapeAttr(setting.max??'')}" step="${escapeAttr(setting.step??(setting.type==='integer'?1:'any'))}"`:'';
    const placeholder=!value&&setting.default!=null?` placeholder="${escapeAttr(setting.default)}"`:'';
    return `<label>${escapeHtml(label)}</label><input id="${id}" type="${type}" value="${escapeAttr(value??'')}"${attrs}${placeholder}${disabled}>`;
  }
  function nativeSettingsHtml(model,params){
    const settings=model?.native_settings||[];
    if(!settings.length)return '<div class="hint">This model has no additional native controls exposed by Wan2GP.</div>';
    return `<details open><summary>Model-native settings</summary><div class="native-settings">${settings.map(setting=>nativeSettingHtml(setting,params)).join('')}</div><div class="hint">These controls come from the selected model metadata. Empty values keep Wan2GP defaults.</div></details>`;
  }
  function readNativeSettings(model,params){
    if(!model)return;
    params.settings=params.settings||{};
    for(const setting of model.native_settings||[]){
      const element=$(nativeSettingId(setting.key));if(!element)continue;
      let value=element.value;
      if(value===''){const target=setting.custom?(params.settings.custom_settings||{}):params.settings;delete target[setting.key];continue;}
      if(setting.type==='integer')value=Math.round(Number(value));
      else if(setting.type==='number')value=Number(value);
      const target=setting.custom?(params.settings.custom_settings||(params.settings.custom_settings={})):params.settings;
      target[setting.key]=value;
    }
    if(params.settings.custom_settings&&!Object.keys(params.settings.custom_settings).length)delete params.settings.custom_settings;
  }

  function uid(){return `n${Date.now().toString(36)}${Math.random().toString(36).slice(2,8)}`;}
  function clone(value){return JSON.parse(JSON.stringify(value));}
  function emit(action){
    window.parent.postMessage({type:'WAN2GP_WORKFLOWS_V2', action, graph:clone(graph)}, '*');
  }
  function incomingEdge(nodeId,port){return graph.edges.find(edge=>edge.target?.node===nodeId&&edge.target?.port===port);}
  function groupRect(group){
    const nodes=(group.node_ids||[]).map(nodeById).filter(Boolean);if(!nodes.length)return null;
    const pad=18, header=27, width=218, height=160;
    const minX=Math.min(...nodes.map(node=>node.position.x)), minY=Math.min(...nodes.map(node=>node.position.y));
    const maxX=Math.max(...nodes.map(node=>node.position.x+width)), maxY=Math.max(...nodes.map(node=>node.position.y+height));
    return {x:minX-pad,y:minY-pad-header,w:maxX-minX+pad*2,h:maxY-minY+pad*2+header};
  }
  function updateGroupGeometry(){graph.groups.forEach(group=>{const element=document.querySelector(`[data-group-id="${CSS.escape(group.id)}"]`),rect=groupRect(group);if(!element||!rect)return;element.style.left=`${rect.x}px`;element.style.top=`${rect.y}px`;element.style.width=`${rect.w}px`;element.style.height=`${rect.h}px`;});}
  function inputOverrideHint(node,port,label){return incomingEdge(node.id,port)?`<div class="hint connected-hint">${escapeHtml(label||port)} comes from a connected node. This inspector value is only a fallback.</div>`:'';}
  function parseLoraMultipliers(value){return String(value||'').split(/[|;,\s]+/).map(Number).filter(value=>Number.isFinite(value));}
  function loraValues(params){
    const value=params?.activated_loras;
    if(Array.isArray(value))return value.map(item=>String(item||'').trim()).filter(Boolean);
    return value?String(value).split(/\r?\n|[|,]/).map(item=>item.trim()).filter(Boolean):[];
  }
  function loraState(params,modelType){
    const stored=params?.loras_by_model?.[modelType];
    if(stored&&typeof stored==='object')return {activated_loras:loraValues(stored),loras_multipliers:String(stored.loras_multipliers||'')};
    if(params?.loras_by_model&&modelType)return {activated_loras:[],loras_multipliers:''};
    return {activated_loras:loraValues(params),loras_multipliers:String(params?.loras_multipliers||'')};
  }
  function saveLoraState(params,modelType,values,multipliers){
    if(!modelType)return;
    params.loras_by_model=params.loras_by_model||{};
    params.loras_by_model[modelType]={activated_loras:[...values],loras_multipliers:String(multipliers||'')};
  }
  function readLoraControls(host){
    const checks=Array.from(host.querySelectorAll('[data-lora-name]')).filter(input=>input.checked);
    return {values:checks.map(input=>input.dataset.loraName),multipliers:checks.map(input=>host.querySelector(`[data-lora-strength="${CSS.escape(input.dataset.loraName)}"]`)?.value||'1').join('|')};
  }
  function isRemoteLora(value){return /^https?:\/\//i.test(String(value||''));}
  function loraRowsHtml(loras,selected,multipliers){
    const values=[...loras,...selected.filter(value=>!loras.includes(value))];
    const rows=values.map((value,index)=>{const selectedIndex=selected.indexOf(value);const strength=selectedIndex>=0?(multipliers[selectedIndex]??1):1;const remote=isRemoteLora(value);return `<div class="lora-row${remote?' remote-lora':''}" data-lora-row="${escapeAttr(value.toLowerCase())}"><label><input type="checkbox" data-lora-name="${escapeAttr(value)}" ${selectedIndex>=0?'checked':''}> <span title="${escapeAttr(value)}">${remote?'Remote · ':''}${escapeHtml(value)}</span></label><input class="lora-strength" data-lora-strength="${escapeAttr(value)}" type="number" min="-4" max="4" step="0.05" value="${escapeAttr(strength)}" ${selectedIndex<0?'disabled':''}></div>`;}).join('');
    return `<input id="editLoraSearch" placeholder="Search local and remote LoRAs...">${values.length?`<div class="lora-list">${rows}</div>`:'<div class="hint">No local LoRAs were found for this model yet. You can add a remote URL below.</div>'}<div class="hint">Activate any number of LoRAs and set each force independently.</div><div class="lora-remote-entry"><input id="editLoraUrl" type="url" placeholder="https://huggingface.co/.../resolve/main/model.safetensors"><button type="button" id="editAddLoraUrl">Add URL</button></div><div class="hint">Wan2GP will resolve and download remote LoRAs when the workflow runs. Optional subfolders use the native <code>URL|folder</code> format.</div>`;
  }
  function modelPorts(node){
    const type = node.type;
    const params = node.params || {};
    const model = modelById(params.model_type);
    const caps = model?.capabilities || {};
    const image = caps.image_inputs || {};
    const video = caps.video_inputs || {};
    const audio = caps.audio_inputs || {};
    const output = (caps.main_output || [type.includes('image') ? 'image' : type.includes('audio') ? 'audio' : 'video'])[0];
    const ports = [{name:'prompt', direction:'in', type:'TEXT', optional:true}];
    if(image.start || image.reference || type==='edit_image' || type==='inpaint_image') ports.push({name:'image',direction:'in',type:'IMAGE',optional:!type.startsWith('edit_')&&!type.startsWith('inpaint_')});
    if(image.multiple_references) ports.push({name:'references',label:model?.reference_input_label||'references',direction:'in',type:'IMAGE[]',variadic:true,optional:true});
    if(video.continue || video.reference || video.control || type==='edit_video' || type==='inpaint_video') ports.push({name:'video',direction:'in',type:'VIDEO',optional:!type.startsWith('edit_')&&!type.startsWith('inpaint_')});
    if(video.reference) ports.push({name:'reference_videos',direction:'in',type:'VIDEO[]',variadic:true,optional:true});
    if(audio.prompt || type === 'generate_audio') ports.push({name:'audio',direction:'in',type:'AUDIO',optional:true});
    if(caps.inpainting || type.startsWith('inpaint_')) ports.push({name:'mask',direction:'in',type:String(output).toLowerCase() === 'image' ? 'MASK_IMAGE' : 'MASK_VIDEO'});
    if(image.end) ports.push({name:'end_frame',direction:'in',type:'IMAGE',optional:true});
    if(image.injected_frames) ports.push({name:'frames',direction:'in',type:'IMAGE[]',variadic:true,optional:true});
    ports.push({name:'output',direction:'out',type:String(output || 'video').toUpperCase()});
    return ports;
  }
  function portsFor(node){
    if(['generate_image','edit_image','inpaint_image','generate_video','edit_video','inpaint_video','generate_audio'].includes(node.type)) return modelPorts(node);
    const map = {
      input_image:[['image','out','IMAGE']], input_video:[['video','out','VIDEO']], input_audio:[['audio','out','AUDIO']], input_mask_image:[['mask','out','MASK_IMAGE']], input_mask_video:[['mask','out','MASK_VIDEO']], load_media:[['media','out','MEDIA']], text:[['text','out','TEXT']],
      mask_editor:[['source','in','IMAGE'],['mask','out','MASK_IMAGE']], magic_mask:[['source','in','MEDIA'],['mask_image','out','MASK_IMAGE'],['mask_video','out','MASK_VIDEO']], ai_analyze:[['images','in','IMAGE[]'],['instruction','in','TEXT'],['text','out','TEXT']], last_frame:[['video','in','VIDEO'],['image','out','IMAGE']], extract_frame:[['video','in','VIDEO'],['image','out','IMAGE']], image_sequence:[['images','in','IMAGE[]'],['video','out','VIDEO']], video_probe:[['video','in','VIDEO'],['metadata','out','TEXT']], postprocess:[['media','in','MEDIA'],['references','in','IMAGE[]'],['audio','in','AUDIO'],['output','out','MEDIA']],
      ffmpeg_trim:[['video','in','VIDEO'],['output','out','VIDEO']], ffmpeg_concat:[['videos','in','VIDEO[]'],['output','out','VIDEO']], ffmpeg_mux_audio:[['video','in','VIDEO'],['audio','in','AUDIO'],['output','out','VIDEO']], ffmpeg_remove_audio:[['video','in','VIDEO'],['output','out','VIDEO']], ffmpeg_transcode:[['media','in','MEDIA'],['output','out','MEDIA']], ffmpeg_resize:[['video','in','VIDEO'],['output','out','VIDEO']], ffmpeg_fps:[['video','in','VIDEO'],['output','out','VIDEO']], ffmpeg_normalize:[['video','in','VIDEO'],['output','out','VIDEO']], ffmpeg_export:[['video','in','VIDEO'],['output','out','VIDEO']]
    };
    return (map[node.type] || []).map(([name,direction,type]) => ({name,direction,type,optional:true,variadic:type.endsWith('[]')}));
  }
  function portDef(node, name){return portsFor(node).find(port => port.name === name);}
  function findPortElement(nodeId, name){return document.querySelector(`[data-node="${nodeId}"][data-port="${name}"]`);}
  function center(element){const a=element.getBoundingClientRect(), b=$("viewport").getBoundingClientRect();return {x:a.left-b.left+a.width/2,y:a.top-b.top+a.height/2};}
  function previewFor(node){
    if(!['input_image','input_video','input_audio','input_mask_image','input_mask_video','load_media'].includes(node.type))return null;
    return runtimePreviews[Number(node.params?.slot??0)]||null;
  }
  function previewHtml(node){
    const preview=previewFor(node);if(!preview)return '';
    if(preview.kind==='image'||preview.kind==='video')return `<img class="media-preview" src="${escapeAttr(preview.src)}" alt="${escapeAttr(preview.name||'Preview')}">`;
    if(preview.kind==='audio')return `<div class="media-preview audio-preview">♫ ${escapeHtml(preview.name||'Audio')}</div>`;
    return `<div class="media-preview file-preview">${escapeHtml(preview.name||'File')}</div>`;
  }
  function outputHtml(node){
    if(['input_image','input_video','input_audio','input_mask_image','input_mask_video','load_media'].includes(node.type))return '';
    const output=executionState.nodes?.[node.id]?.output;
    if(!output)return '';
    if(output.kind==='image'||output.kind==='video')return `<div class="output-label">${escapeHtml(output.port||'output')}</div><img class="media-preview node-output" src="${escapeAttr(output.src)}" alt="Node output">`;
    if(output.kind==='text')return `<div class="output-label">${escapeHtml(output.port||'output')}</div><pre class="node-output-text">${escapeHtml(output.text||'')}</pre>`;
    if(output.kind==='audio')return `<div class="output-label">${escapeHtml(output.port||'output')}</div><div class="media-preview audio-preview">♫ ${escapeHtml(output.name||'Audio output')}</div>`;
    return '';
  }
  function executionClass(node){
    const state=executionState.nodes?.[node.id]?.state||'';
    return state ? ` execution-${typeClass(state)}` : '';
  }

  function renderPalette(){
    const search = ($('nodeSearch').value || '').toLowerCase();
    const items = catalog.nodes.filter(item => !search || `${item.label} ${item.category}`.toLowerCase().includes(search));
    const blocks=(catalog.blocks||[]).filter(item=>!search||`${item.name} ${item.description}`.toLowerCase().includes(search));
    $('palette').innerHTML = items.map(item => `<button data-action="add" data-node="${item.type}">${item.label}</button>`).join('') + (blocks.length?`<h4 class="palette-section">Reusable blocks</h4>${blocks.map(item=>{const pending=pendingDeleteBlockId===item.id;return `<div class="block-row${pending?' confirm-delete':''}"><button class="block-button" data-action="add-block" data-block-id="${escapeAttr(item.id)}">${escapeHtml(item.name)}</button><button class="block-delete" data-action="delete-block" data-block-id="${escapeAttr(item.id)}" title="${pending?'Confirm delete':'Delete reusable block'}">${pending?'✓':'×'}</button></div>`;}).join('')}<div class="block-hint">Click × twice to delete a block.</div>`:'');
  }
  function render(){
    const canvas=$('canvas'), svg=$('edges'); canvas.innerHTML=''; svg.innerHTML=''; $('empty').style.display=graph.nodes.length?'none':'block';
    graph.groups.forEach(group=>{
      const rect=groupRect(group);if(!rect)return;
      const el=document.createElement('section');el.className=`graph-group ${group.id===selectedGroup?'selected':''}${group.enabled===false?' disabled':''}`;el.dataset.groupId=group.id;el.style.left=`${rect.x}px`;el.style.top=`${rect.y}px`;el.style.width=`${rect.w}px`;el.style.height=`${rect.h}px`;el.style.borderColor=group.color||'#395b78';el.style.background=`${group.color||'#395b78'}24`;
      el.innerHTML=`<div class="group-head" style="background:${escapeAttr(group.color||'#395b78')}66"><span>${escapeHtml(group.title||'Group')}</span><small>${group.enabled===false?'OFF':'GROUP'}</small><button class="group-toggle" data-group-action="toggle" title="${group.enabled===false?'Enable':'Disable'} group">${group.enabled===false?'▶':'⏸'}</button></div>`;
      canvas.appendChild(el);
      el.querySelector('.group-head').addEventListener('pointerdown',event=>startGroupDrag(event,group,el));
      el.addEventListener('click',event=>{if(!event.target.closest('.group-toggle')){selectGroup(group.id);render();}});
      el.querySelector('.group-toggle').addEventListener('click',event=>{event.stopPropagation();group.enabled=group.enabled===false;render();});
    });
    graph.nodes.forEach(node => {
      const el=document.createElement('article'); el.className=`node ${selectedNodes.includes(node.id)?'selected':''}${node.enabled===false?' disabled':''}${executionClass(node)}`; el.dataset.nodeId=node.id; el.style.left=`${node.position.x}px`; el.style.top=`${node.position.y}px`;
      const params=node.params||{}; const model=modelById(params.model_type);
      const inPorts=portsFor(node).filter(port=>port.direction==='in'), outPorts=portsFor(node).filter(port=>port.direction==='out');
      const summary=node.type.startsWith('generate')||node.type.startsWith('edit_')||node.type.startsWith('inpaint_') ? (model?.name || params.model_type || 'Select model') : (params.process_id || params.process_type || node.type);
      const state=executionState.nodes?.[node.id]?.state||'';
      el.innerHTML=`<div class="node-head"><span>${escapeHtml(node.title)}</span><small>${escapeHtml(state||node.type)}</small><button class="node-toggle" data-node-action="toggle" title="${node.enabled===false?'Enable':'Disable'} node">${node.enabled===false?'▶':'⏸'}</button></div><div class="node-body"><span class="badge">${escapeHtml(summary)}</span>${previewHtml(node)}${outputHtml(node)}${params.prompt?`<div>${escapeHtml(String(params.prompt).slice(0,80))}</div>`:''}${params.warning?`<div class="warning">${escapeHtml(params.warning)}</div>`:''}</div><div class="ports"><div class="ports-col">${inPorts.map(port=>portHtml(node,port)).join('')}</div><div class="ports-col">${outPorts.map(port=>portHtml(node,port)).join('')}</div></div>`;
      canvas.appendChild(el);
      el.addEventListener('click', event=>{if(!event.target.closest('.port,.node-toggle')){selectNode(node.id,event.shiftKey);selectedEdge=null;render();}});
      el.querySelector('.node-head').addEventListener('pointerdown', event=>startNodeDrag(event,node,el));
      el.querySelector('.node-toggle').addEventListener('click',event=>{event.stopPropagation();node.enabled=node.enabled===false;render();});
      el.querySelectorAll('.port').forEach(port=>{
        port.addEventListener('pointerdown',event=>{port.dataset.pointerHandled='1';portDown(event,node,port);});
        port.addEventListener('mousedown',event=>{if(!port.dataset.pointerHandled){port.dataset.pointerHandled='1';portDown(event,node,port);}});
        // Some embedded browsers deliver the click but omit pointerdown on
        // sandboxed srcdoc content. Keep click as a deduplicated fallback.
        port.addEventListener('click',event=>{if(port.dataset.pointerHandled){delete port.dataset.pointerHandled;return;}portDown(event,node,port);});
      });
    });
    requestAnimationFrame(drawEdges); renderInspector();
  }
  function portHtml(node,port){const c=typeClass(port.type);return `<div class="port ${port.direction}" data-node="${escapeAttr(node.id)}" data-port="${escapeAttr(port.name)}" data-type="${escapeAttr(port.type)}"><span class="dot ${c}"></span><span>${escapeHtml(port.label||port.name)}</span></div>`;}
  function drawEdges(){
    const svg=$('edges'), viewport=$('viewport'), rect=viewport.getBoundingClientRect(); svg.innerHTML=''; svg.setAttribute('viewBox',`0 0 ${rect.width} ${rect.height}`);
    graph.edges.forEach(edge=>{const a=findPortElement(edge.source.node,edge.source.port),b=findPortElement(edge.target.node,edge.target.port);if(!a||!b)return;const p1=center(a),p2=center(b),dx=Math.max(45,Math.abs(p2.x-p1.x)*.45);const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',`M ${p1.x} ${p1.y} C ${p1.x+dx} ${p1.y}, ${p2.x-dx} ${p2.y}, ${p2.x} ${p2.y}`);path.setAttribute('class',`edge ${typeClass(edge.type)}${edge.id===selectedEdge?' selected':''}`);path.style.pointerEvents='stroke';path.dataset.edgeId=edge.id;path.addEventListener('pointerdown',event=>{event.stopPropagation();selectedEdge=edge.id;selected=null;pending=null;render();});svg.appendChild(path);});
  }
  function startNodeDrag(event,node,element){if(event.target.closest('.node-toggle'))return;event.preventDefault();if(!selectedNodes.includes(node.id))selectNode(node.id,event.shiftKey);selectedEdge=null;const moving=selectedNodes.map(id=>nodeById(id)).filter(Boolean);dragging={nodes:moving,startX:event.clientX,startY:event.clientY,positions:moving.map(item=>({node:item,x:item.position.x,y:item.position.y}))};element.setPointerCapture?.(event.pointerId);renderInspector();}
  function startGroupDrag(event,group,element){if(event.target.closest('.group-toggle'))return;event.preventDefault();selectGroup(group.id);const moving=(group.node_ids||[]).map(nodeById).filter(Boolean);dragging={nodes:moving,startX:event.clientX,startY:event.clientY,positions:moving.map(item=>({node:item,x:item.position.x,y:item.position.y}))};element.setPointerCapture?.(event.pointerId);renderInspector();}
  function portDown(event,node,port){event.stopPropagation();selectedEdge=null;const name=port.dataset.port,def=portDef(node,name);if(def.direction==='out'){pending={node:node.id,port:name,type:def.type};selected=node.id;return;}if(pending){const target=def;if(compatible(pending.type,target.type)){graph.edges=graph.edges.filter(edge=>!(edge.target.node===node.id&&edge.target.port===name&&!target.variadic));graph.edges.push({id:uid(),source:{node:pending.node,port:pending.port},target:{node:node.id,port:name},type:pending.type});}pending=null;render();}}
  function compatible(a,b){a=portBase(a);b=portBase(b);return a===b||a==='MEDIA'||b==='MEDIA';}
  function renderLegacyInspector(){const node=nodeById(selected), host=$('inspector'), empty=$('inspectorEmpty');if(!node){empty.style.display='block';host.innerHTML='';return;}empty.style.display='none';const p=node.params||{};const models=catalog.models.filter(model=>{const output=String((model.capabilities?.main_output||[])[0]||'').toLowerCase();return node.type.includes('image')?output==='image':node.type.includes('audio')?output==='audio':node.type.includes('video')?output==='video':true;});host.innerHTML=`<label>Title</label><input id="editTitle" value="${escapeAttr(node.title)}"><label>Model / finetune</label><select id="editModel"><option value="">Select model</option>${models.map(model=>`<option value="${escapeAttr(model.model_type)}" ${model.model_type===p.model_type?'selected':''}>${escapeHtml(model.name||model.model_type)}</option>`).join('')}</select><label>Prompt</label><textarea id="editPrompt">${escapeHtml(p.prompt||'')}</textarea><label>Negative prompt</label><textarea id="editNegative">${escapeHtml(p.negative_prompt||'')}</textarea><div class="row"><div><label>Seed</label><input id="editSeed" type="number" value="${escapeAttr(p.seed??-1)}"></div><div><label>Frames</label><input id="editFrames" type="number" value="${escapeAttr(p.video_length??81)}"></div></div><label>Processor / method</label><select id="editProcess"><option value="">Select</option>${(catalog.processes||[]).map(item=>`<option value="${escapeAttr(item.id)}" ${item.id===p.process_id?'selected':''}>${escapeHtml(item.label)}</option>`).join('')}</select><label>Native settings JSON</label><textarea id="editRaw">${escapeHtml(JSON.stringify(p.settings||{},null,2))}</textarea><div class="hint">Advanced model properties are preserved here even when a model adds a field the editor does not know yet.</div>`;
    const update=()=>{node.title=$('editTitle').value;node.params=p; p.model_type=$('editModel').value; p.prompt=$('editPrompt').value;p.negative_prompt=$('editNegative').value;p.seed=Number($('editSeed').value);p.video_length=Number($('editFrames').value);p.process_id=$('editProcess').value;const process=(catalog.processes||[]).find(item=>item.id===p.process_id);if(process){p.process_type=process.type;p.method=process.id;}else if(!p.process_id){delete p.process_type;delete p.method;}try{p.settings=JSON.parse($('editRaw').value||'{}');delete p.warning;}catch(error){p.warning='Native settings JSON is invalid.';}render();};
    ['editTitle','editModel','editPrompt','editNegative','editSeed','editFrames','editProcess','editRaw'].forEach(id=>$(id)?.addEventListener('change',update));
  }
  function processById(id){return (catalog.processes||[]).find(item=>item.id===id);}
  function processParameterValue(params, parameter){
    const values=params.parameters||{};
    if(Object.prototype.hasOwnProperty.call(values,parameter.name))return values[parameter.name];
    if(Object.prototype.hasOwnProperty.call(params,parameter.name))return params[parameter.name];
    if(Object.prototype.hasOwnProperty.call(parameter,'default'))return parameter.default;
    if(Array.isArray(parameter.enum)&&parameter.enum.length)return parameter.enum[0];
    return parameter.type==='array'?[]:parameter.type==='boolean'?false:'';
  }
  function processParameterHtml(parameter,value){
    const id=`proc_${String(parameter.name).replace(/[^A-Za-z0-9_]/g,'_')}`;
    const label=parameter.label||parameter.name.replaceAll('_',' ');
    const description=parameter.description?`<div class="hint">${escapeHtml(parameter.description)}</div>`:'';
    if(parameter.type==='boolean')return `<label><input id="${id}" type="checkbox" ${value?'checked':''}> ${escapeHtml(label)}</label>${description}`;
    if(Array.isArray(parameter.enum)&&parameter.enum.length)return `<label>${escapeHtml(label)}</label><select id="${id}">${parameter.enum.map(option=>`<option value="${escapeAttr(option)}" ${String(option)===String(value)?'selected':''}>${escapeHtml(option)}</option>`).join('')}</select>${description}`;
    if(parameter.type==='array')return `<label>${escapeHtml(label)}</label><textarea id="${id}" placeholder="Connect the IMAGE[] port for media references">${escapeHtml(Array.isArray(value)?value.join('\n'):value||'')}</textarea>${description}`;
    const type=parameter.type==='integer'||parameter.type==='number'?'number':'text';
    const attrs=type==='number'?` min="${escapeAttr(parameter.minimum??'')}" max="${escapeAttr(parameter.maximum??'')}" step="${escapeAttr(parameter.step??(parameter.type==='integer'?1:'any'))}"`:'';
    return `<label>${escapeHtml(label)}</label><input id="${id}" type="${type}" value="${escapeAttr(value??'')}"${attrs}>${description}`;
  }
  function processParametersHtml(process,params){return (process?.parameters||[]).map(parameter=>processParameterHtml(parameter,processParameterValue(params,parameter))).join('');}

  function renderInspector(){
    const node=nodeById(selected), host=$('inspector'), empty=$('inspectorEmpty');
    const group=groupById(selectedGroup);
    if(group){
      empty.style.display='none';
      host.innerHTML=`<h4>Reusable group</h4><label>Title</label><input id="editGroupTitle" value="${escapeAttr(group.title||'Group')}"><label>Color</label><input id="editGroupColor" type="color" value="${escapeAttr(group.color||'#395b78')}"><label class="check-row"><input id="editGroupEnabled" type="checkbox" ${group.enabled===false?'':'checked'}> Enabled</label><div class="hint">${(group.node_ids||[]).length} node(s). When disabled, a compatible single-input/single-output chain is bypassed automatically.</div><div class="inspector-actions"><button data-action="save-block" class="primary">Save as reusable block</button><button data-action="ungroup">Ungroup</button></div>`;
      const update=()=>{group.title=$('editGroupTitle').value.trim()||'Group';group.color=$('editGroupColor').value;group.enabled=$('editGroupEnabled').checked;render();};
      ['editGroupTitle','editGroupColor','editGroupEnabled'].forEach(id=>$(id)?.addEventListener('change',update));return;
    }
    if(!node){
      empty.style.display='none';
      const settings=graph.settings||{};
      const ratios=catalog.aspect_ratios||[{label:'Initial input',value:'source'}];
      host.innerHTML=`<h4>Workflow settings</h4><label>Default aspect ratio</label><select id="editWorkflowAspect">${ratios.map(item=>`<option value="${escapeAttr(item.value)}" ${item.value===(settings.aspect_ratio||'source')?'selected':''}>${escapeHtml(item.label)}</option>`).join('')}</select><div class="hint">Generation nodes use this ratio unless they define an override in their inspector.</div>`;
      $('editWorkflowAspect').addEventListener('change',event=>{graph.settings=graph.settings||{};graph.settings.aspect_ratio=event.target.value;});
      return;
    }
    empty.style.display='none';
    const p=node.params||{};
    const bind=(ids,update)=>ids.forEach(id=>$(id)?.addEventListener('change',update));
    const rawSettings=()=>{try{p.settings=JSON.parse($('editRaw').value||'{}');delete p.warning;}catch(error){p.warning='Native settings JSON is invalid.';}};
    const title=`<label>Title</label><input id="editTitle" value="${escapeAttr(node.title)}">`;

    if(node.type==='ai_analyze'){
      const aiModels=catalog.ai_models||[];
      const instructionConnected=Boolean(incomingEdge(node.id,'instruction'));
      host.innerHTML=title+`<label>Instruction for AI</label><textarea id="editInstruction" placeholder="Describe the image, extract details, or return a prompt..." ${instructionConnected?'disabled':''}>${escapeHtml(p.instruction||'')}</textarea>${inputOverrideHint(node,'instruction','Instruction')}<label>AI model</label><select id="editAiModel">${aiModels.map(item=>`<option value="${item.id}" ${Number(p.model??3)===Number(item.id)?'selected':''}>${escapeHtml(item.label)}</option>`).join('')}</select><div class="row"><div><label>Max tokens</label><input id="editMaxTokens" type="number" min="1" value="${escapeAttr(p.max_tokens??192)}"></div><div><label>Output variable</label><input id="editOutputName" value="${escapeAttr(p.output_name||'analysis')}"></div></div><div class="hint">Inputs: images and optional instruction text. Output: TEXT and a named variable (usable as $analysis or your chosen name). This node does not use Wan2GP generation models, frames, seed, or processors.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.instruction=$('editInstruction').value;p.model=Number($('editAiModel').value||3);p.max_tokens=Number($('editMaxTokens').value||192);p.output_name=$('editOutputName').value.trim()||'analysis';render();};
      bind(['editTitle','editInstruction','editAiModel','editMaxTokens','editOutputName'],update);return;
    }

    if(['input_image','input_video','input_audio','input_mask_image','input_mask_video','load_media'].includes(node.type)){
      host.innerHTML=title+`<label>Runtime input slot</label><input id="editSlot" type="number" min="0" value="${escapeAttr(p.slot??0)}"><div class="hint">This node reads the corresponding file from the runtime input list.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.slot=Math.max(0,Number($('editSlot').value||0));render();};
      bind(['editTitle','editSlot'],update);return;
    }

    if(node.type==='mask_editor'){
      host.innerHTML=title+`<label>Mask file path (optional)</label><input id="editMaskPath" value="${escapeAttr(p.mask_path||'')}" placeholder="Use the existing mask editor or provide a mask file"><div class="hint">Output: MASK_IMAGE. Connect it only to an image inpainting node.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.mask_path=$('editMaskPath').value.trim();render();};
      bind(['editTitle','editMaskPath'],update);return;
    }

    if(node.type==='magic_mask'){
      const maxObjects=String(p.max_objects??'all');
      host.innerHTML=title+`<label>Objects / persons to track</label><textarea id="editMagicKeywords" placeholder="person, car, sky">${escapeHtml(p.keywords||'')}</textarea><label class="check-row"><input id="editMagicNegative" type="checkbox" ${p.negative_mask?'checked':''}> Negative mask</label><label>Max objects</label><select id="editMagicMaxObjects"><option value="all" ${maxObjects==='all'?'selected':''}>All</option>${[1,2,3,4,5].map(value=>`<option value="${value}" ${maxObjects===String(value)?'selected':''}>${value}</option>`).join('')}</select><label>Max video time (seconds)</label><input id="editMagicMaxTime" type="number" min="0.1" step="0.1" value="${escapeAttr(p.max_time??'')}"><div class="hint">Connect IMAGE or VIDEO to source. Use mask_image for image inpainting and mask_video for video inpainting. The node downloads/uses Wan2GP's managed Magic Mask assets at runtime.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.keywords=$('editMagicKeywords').value;p.negative_mask=$('editMagicNegative').checked;p.max_objects=$('editMagicMaxObjects').value;p.max_time=$('editMagicMaxTime').value?Number($('editMagicMaxTime').value):'';render();};
      bind(['editTitle','editMagicKeywords','editMagicNegative','editMagicMaxObjects','editMagicMaxTime'],update);return;
    }

    if(node.type==='last_frame' || node.type==='extract_frame'){
      host.innerHTML=title+(node.type==='extract_frame'?`<label>Frame number</label><input id="editFrame" type="number" min="1" value="${escapeAttr(p.frame??1)}">`:'')+`<div class="hint">Input: VIDEO. Output: IMAGE.</div>`;
      const update=()=>{node.title=$('editTitle').value;if($('editFrame'))p.frame=Math.max(1,Number($('editFrame').value||1));render();};
      bind(['editTitle','editFrame'],update);return;
    }

    if(node.type==='image_sequence'){
      host.innerHTML=title+`<label>FPS</label><input id="editSequenceFps" type="number" min="0.1" step="0.1" value="${escapeAttr(p.fps??24)}"><div class="hint">Connect one or more IMAGE inputs. The node encodes them in connection order into a video.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.fps=Math.max(0.1,Number($('editSequenceFps').value||24));render();};
      bind(['editTitle','editSequenceFps'],update);return;
    }

    if(node.type.startsWith('ffmpeg_')){
      const fields={
        ffmpeg_trim:`<label>Start (seconds)</label><input id="editStart" type="number" min="0" step="0.01" value="${escapeAttr(p.start??0)}"><label>Duration (seconds)</label><input id="editDuration" type="number" min="0.01" step="0.01" value="${escapeAttr(p.duration??5)}">`,
        ffmpeg_resize:`<div class="row"><div><label>Width</label><input id="editWidth" type="number" min="1" value="${escapeAttr(p.width??832)}"></div><div><label>Height</label><input id="editHeight" type="number" min="1" value="${escapeAttr(p.height??480)}"></div></div>`,
        ffmpeg_fps:`<label>FPS</label><input id="editFps" type="number" min="1" step="0.01" value="${escapeAttr(p.fps??24)}">`
      }[node.type]||'';
      host.innerHTML=title+fields+`<div class="hint">FFmpeg operation: ${escapeHtml(node.type.replace('ffmpeg_',''))}. Paths and commands are managed by Wan2GP.</div>`;
      const update=()=>{node.title=$('editTitle').value;if($('editStart'))p.start=Number($('editStart').value||0);if($('editDuration'))p.duration=Number($('editDuration').value||5);if($('editWidth'))p.width=Math.max(1,Number($('editWidth').value||832));if($('editHeight'))p.height=Math.max(1,Number($('editHeight').value||480));if($('editFps'))p.fps=Number($('editFps').value||24);render();};
      bind(['editTitle','editStart','editDuration','editWidth','editHeight','editFps'],update);return;
    }

    if(node.type==='text'){
      host.innerHTML=title+`<label>Text</label><textarea id="editText" placeholder="Prompt or text variable...">${escapeHtml(p.text||p.value||'')}</textarea><div class="hint">Output: TEXT. Connect it to a prompt or instruction input.</div>`;
      const update=()=>{node.title=$('editTitle').value;p.text=$('editText').value;delete p.value;render();};
      bind(['editTitle','editText'],update);return;
    }

    if(node.type==='postprocess'){
      const renderProcessInspector=()=>{
        const process=processById(p.process_id);
        host.innerHTML=title+`<label>Processor / method</label><select id="editProcess"><option value="">Select processor</option>${(catalog.processes||[]).map(item=>`<option value="${escapeAttr(item.id)}" ${item.id===p.process_id?'selected':''}>${escapeHtml(item.label)}${item.status==='disabled'?' (disabled)':''}</option>`).join('')}</select>${process?`<div class="hint">${escapeHtml(process.description||'')} ${process.media?.length?`Media: ${escapeHtml(process.media.join(', '))}.`:''}</div>${processParametersHtml(process,p)}`:'<div class="hint">Select a processor to expose its native controls.</div>'}<details><summary>Advanced native settings</summary><textarea id="editRaw">${escapeHtml(JSON.stringify(p.settings||{},null,2))}</textarea></details>`;
        const update=()=>{
          node.title=$('editTitle').value;
          const nextId=$('editProcess').value;
          if(nextId!==p.process_id){p.process_id=nextId;p.parameters={};}
          const selectedProcess=processById(p.process_id);
          if(selectedProcess){p.process_type=selectedProcess.type;p.method=selectedProcess.id;p.parameters=p.parameters||{};for(const parameter of selectedProcess.parameters||[]){const id=`proc_${String(parameter.name).replace(/[^A-Za-z0-9_]/g,'_')}`, element=$(id);if(!element)continue;let value=parameter.type==='boolean'?element.checked:parameter.type==='array'?element.value.split(/\r?\n/).map(item=>item.trim()).filter(Boolean):element.value;if(parameter.type==='number')value=Number(value);if(parameter.type==='integer')value=Math.round(Number(value));p.parameters[parameter.name]=value;}}
          else{delete p.process_type;delete p.method;p.parameters={};}
          if($('editRaw'))rawSettings();render();
        };
        bind(['editTitle','editProcess','editRaw',...(process?.parameters||[]).map(parameter=>`proc_${String(parameter.name).replace(/[^A-Za-z0-9_]/g,'_')}`)],update);
      };
      renderProcessInspector();return;
    }

    if(['generate_image','edit_image','inpaint_image','generate_video','edit_video','inpaint_video','generate_audio'].includes(node.type)){
      const output=node.type.includes('audio')?'audio':node.type.includes('image')?'image':'video';
      const model=modelById(p.model_type), models=modelChoicesFor(node), caps=model?.capabilities||{}, resolutions=model?.resolution_choices||[], resolutionTiers=model?.resolution_tiers||Array.from(new Set(resolutions.map(item=>item.tier))).map(value=>({label:value,value})), currentTier=p.resolution_tier||resolutions.find(item=>item.value===p.resolution)?.tier||'720p', ratios=catalog.aspect_ratios||[], workflowRatio=graph.settings?.aspect_ratio||'source', workflowRatioLabel=ratios.find(item=>item.value===workflowRatio)?.label||workflowRatio;
      const imageInputs=caps.image_inputs||{}, videoInputs=caps.video_inputs||{};
      const sourceModes=[['auto','Automatic']];
      if(imageInputs.start)sourceModes.push(['start_image','Start image']);
      if(imageInputs.reference)sourceModes.push(['reference_image','Reference image']);
      if(videoInputs.continue)sourceModes.push(['continue_video','Continue video']);
      if(videoInputs.reference)sourceModes.push(['reference_video','Reference video']);
      if(videoInputs.control)sourceModes.push(['control_video','Control video']);
      if(caps.text_only)sourceModes.push(['none','Text only']);
      const resolutionField=output==='audio'?'':`<label>Resolution</label><select id="editResolution"><option value="">Automatic</option>${resolutionTiers.map(item=>`<option value="${escapeAttr(item.value)}" ${String(item.value).toLowerCase()===String(currentTier).toLowerCase()?'selected':''}>${escapeHtml(item.label)}</option>`).join('')}</select><label>Aspect ratio</label><select id="editAspectRatio"><option value="" ${!p.aspect_ratio?'selected':''}>Workflow default (${escapeHtml(workflowRatioLabel)})</option>${ratios.map(item=>`<option value="${escapeAttr(item.value)}" ${item.value===p.aspect_ratio?'selected':''}>${escapeHtml(item.label)}</option>`).join('')}</select>`;
      const sourceModeField=output==='audio'?'':`<label>Input mode</label><select id="editSourceMode">${sourceModes.map(([value,label])=>`<option value="${value}" ${value===(p.source_mode||'auto')?'selected':''}>${label}</option>`).join('')}</select>`;
      const mediaFields=output==='audio'?'':`<label>Negative prompt</label><textarea id="editNegative">${escapeHtml(p.negative_prompt||'')}</textarea>`;
      const frameField=output==='video'?`<label>Frames</label><input id="editFrames" type="number" value="${escapeAttr(p.video_length??81)}">`:'';
      const framePositionsField=output==='video'&&imageInputs.injected_frames?`<label>Injected frame positions</label><input id="editFramePositions" value="${escapeAttr(p.frame_positions??'1')}" placeholder="1 41 L"><div class="hint">One position per connected frame. Use frame numbers, <code>L</code> for the end of a sliding window, or <code>X</code> to skip a window. Example: <code>1 41 L</code>.</div>`:'';
      const loras=model?.lora_supported ? (model.loras||[]) : [];
      const selectedLoraState=loraState(p,p.model_type);
      const selectedLoras=selectedLoraState.activated_loras;
      const loraField=model?.lora_supported?`<label>LoRAs</label>${loraRowsHtml(loras,selectedLoras,parseLoraMultipliers(selectedLoraState.loras_multipliers))}`:'<div class="hint">This model does not support LoRAs.</div>';
      const promptConnected=Boolean(incomingEdge(node.id,'prompt'));
      const maskHint=node.type.startsWith('inpaint_')?'Mask input is required for this inpainting node.':'When connected, MASK switches this image/video generation into Wan2GP native inpainting; without it, generation uses the full canvas.';
      host.innerHTML=title+`<label>Model / finetune</label><select id="editModel"><option value="">Select compatible model</option>${models.map(model=>`<option value="${escapeAttr(model.model_type)}" ${model.model_type===p.model_type?'selected':''}>${escapeHtml(modelLabel(model))}</option>`).join('')}</select><label>Prompt</label><textarea id="editPrompt" ${promptConnected?'disabled':''}>${escapeHtml(p.prompt||'')}</textarea>${inputOverrideHint(node,'prompt','Prompt')}${mediaFields}${sourceModeField}${resolutionField}${loraField}<div class="row"><div><label>Seed</label><input id="editSeed" type="number" value="${escapeAttr(p.seed??-1)}"></div><div>${frameField}</div></div>${framePositionsField}<label>Steps</label><input id="editSteps" type="number" min="1" value="${escapeAttr(p.steps??p.num_inference_steps??20)}">${nativeSettingsHtml(model,p)}<details><summary>Advanced native JSON</summary><textarea id="editRaw">${escapeHtml(JSON.stringify(p.settings||{},null,2))}</textarea></details><div class="hint">${maskHint} The model list, ports, ranges and native controls are filtered from Wan2GP metadata.</div>`;
      const nativeControls=(model?.native_settings||[]).map(setting=>nativeSettingId(setting.key));
      const update=()=>{node.title=$('editTitle').value;const previousModel=p.model_type;const previousLoras=readLoraControls(host);if(previousModel)saveLoraState(p,previousModel,previousLoras.values,previousLoras.multipliers);p.model_type=$('editModel').value;const modelChanged=previousModel!==p.model_type;if(modelChanged){p.settings_by_model=p.settings_by_model||{};if(previousModel)p.settings_by_model[previousModel]=clone(p.settings||{});p.settings=clone(p.settings_by_model[p.model_type]||{});const nextLoras=loraState(p,p.model_type);p.activated_loras=nextLoras.activated_loras;p.loras_multipliers=nextLoras.loras_multipliers;delete p.resolution;delete p.resolution_tier;p.source_mode='auto';}if($('editPrompt')&&!promptConnected)p.prompt=$('editPrompt').value;if($('editNegative'))p.negative_prompt=$('editNegative').value;p.seed=Number($('editSeed').value);if($('editFrames'))p.video_length=Number($('editFrames').value);if($('editFramePositions'))p.frame_positions=$('editFramePositions').value.trim();if($('editSteps')){p.steps=Number($('editSteps').value);p.num_inference_steps=p.steps;}if($('editSourceMode'))p.source_mode=$('editSourceMode').value;if($('editResolution')){p.resolution_tier=$('editResolution').value||'720p';delete p.resolution;}if($('editAspectRatio')){const ratio=$('editAspectRatio').value;if(ratio)p.aspect_ratio=ratio;else delete p.aspect_ratio;}if(!modelChanged)readNativeSettings(modelById(p.model_type),p);if(!modelChanged){const currentLoras=readLoraControls(host);p.activated_loras=currentLoras.values;p.loras_multipliers=currentLoras.multipliers;saveLoraState(p,p.model_type,currentLoras.values,currentLoras.multipliers);}render();};
      const addLoraUrl=()=>{const input=$('editLoraUrl'),value=String(input?.value||'').trim();if(!value)return;if(!isRemoteLora(value)){input.setCustomValidity('Use an http:// or https:// LoRA URL.');input.reportValidity();return;}const current=readLoraControls(host);if(!current.values.includes(value))current.values.push(value),current.multipliers=current.multipliers?`${current.multipliers}|1`:'1';p.activated_loras=current.values;p.loras_multipliers=current.multipliers;saveLoraState(p,p.model_type,current.values,current.multipliers);render();};
      bind(['editTitle','editModel','editPrompt','editNegative','editSeed','editFrames','editFramePositions','editSteps','editSourceMode','editResolution','editAspectRatio',...nativeControls],update);$('editRaw')?.addEventListener('change',()=>{rawSettings();render();});host.querySelectorAll('[data-lora-name],[data-lora-strength]').forEach(element=>element.addEventListener('change',event=>{if(event.target.matches('[data-lora-name]')){const strength=host.querySelector(`[data-lora-strength="${CSS.escape(event.target.dataset.loraName)}"]`);if(strength)strength.disabled=!event.target.checked;}update();}));$('editLoraSearch')?.addEventListener('input',event=>{const query=event.target.value.toLowerCase();host.querySelectorAll('[data-lora-row]').forEach(row=>{row.style.display=row.dataset.loraRow.includes(query)?'grid':'none';});});$('editAddLoraUrl')?.addEventListener('click',addLoraUrl);$('editLoraUrl')?.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();addLoraUrl();}});return;
    }

    host.innerHTML=title+`<details><summary>Advanced native settings</summary><textarea id="editRaw">${escapeHtml(JSON.stringify(p.settings||{},null,2))}</textarea></details><div class="hint">${escapeHtml(node.type)} has no model selector. Its ports define the valid connections.</div>`;
    const update=()=>{node.title=$('editTitle').value;rawSettings();render();};
    bind(['editTitle','editRaw'],update);
  }

  function addNode(type){const index=graph.nodes.length;const node={id:uid(),type,title:(catalog.nodes.find(item=>item.type===type)||{}).label||type,position:{x:120+(index%4)*280,y:80+Math.floor(index/4)*190},params:{},ui:{},enabled:true};if(type.startsWith('generate')||type.startsWith('edit_')||type.startsWith('inpaint_'))node.params.settings={};if(type==='text')node.params.text='';if(type==='ai_analyze'){node.params.instruction='';node.params.model=3;node.params.max_tokens=192;node.params.output_name='analysis';}graph.nodes.push(node);selectNode(node.id);selectedEdge=null;render();}
  function createGroup(){const ids=[...selectedNodes];if(!ids.length&&selected)ids.push(selected);if(!ids.length)return;const colors=['#395b78','#6b4f8a','#39785f','#89633d','#82506a'];const group={id:uid(),title:`Group ${graph.groups.length+1}`,color:colors[graph.groups.length%colors.length],node_ids:ids,enabled:true};graph.groups.push(group);selectGroup(group.id);render();}
  function insertBlock(blockId){const block=(catalog.blocks||[]).find(item=>item.id===blockId);if(!block)return;const nodes=block.nodes||[], idMap={};const minX=Math.min(...nodes.map(node=>Number(node.position?.x||0)),0),minY=Math.min(...nodes.map(node=>Number(node.position?.y||0)),0),offsetX=180-minX+(graph.nodes.length%3)*30,offsetY=100-minY+(graph.nodes.length%3)*30;const clones=nodes.map(raw=>{const node=clone(raw);idMap[node.id]=uid();node.id=idMap[node.id];node.position={x:Number(node.position?.x||0)+offsetX,y:Number(node.position?.y||0)+offsetY};node.enabled=node.enabled!==false;return node;});graph.nodes.push(...clones);graph.edges.push(...(block.edges||[]).map(raw=>{const edge=clone(raw);edge.id=uid();edge.source.node=idMap[edge.source.node]||edge.source.node;edge.target.node=idMap[edge.target.node]||edge.target.node;return edge;}));const sourceGroup=(block.groups||[])[0];if(sourceGroup){const group=clone(sourceGroup);group.id=uid();group.node_ids=(group.node_ids||[]).map(id=>idMap[id]).filter(Boolean);graph.groups.push(group);selectGroup(group.id);}else{selectedNodes=clones.map(node=>node.id);selected=selectedNodes[0]||null;}selectedEdge=null;render();}
  function saveSelectedBlock(){const group=groupById(selectedGroup);if(!group)return;graph.ui=graph.ui||{};graph.ui.selected_group_id=group.id;emit('save_block');}
  function deleteBlock(blockId){const block=(catalog.blocks||[]).find(item=>item.id===blockId);if(!block)return;if(pendingDeleteBlockId!==blockId){pendingDeleteBlockId=blockId;renderPalette();return;}pendingDeleteBlockId=null;graph.ui=graph.ui||{};graph.ui.selected_block_id=blockId;emit('delete_block');}
  function ungroup(){if(!selectedGroup)return;graph.groups=graph.groups.filter(group=>group.id!==selectedGroup);selectedGroup=null;render();}
  function deleteSelection(){if(selectedEdge){graph.edges=graph.edges.filter(edge=>edge.id!==selectedEdge);selectedEdge=null;render();return;}const ids=new Set(selectedNodes.length?selectedNodes:(selected?[selected]:[]));if(!ids.size)return;graph.nodes=graph.nodes.filter(node=>!ids.has(node.id));graph.edges=graph.edges.filter(edge=>!ids.has(edge.source.node)&&!ids.has(edge.target.node));graph.groups=graph.groups.map(group=>({...group,node_ids:(group.node_ids||[]).filter(id=>!ids.has(id))})).filter(group=>group.node_ids.length);selected=null;selectedNodes=[];render();}
  function fit(){if(!graph.nodes.length)return;const xs=graph.nodes.map(n=>n.position.x),ys=graph.nodes.map(n=>n.position.y);pan={x:Math.max(0,120-Math.min(...xs)),y:Math.max(0,80-Math.min(...ys))};scale=1;applyView();}
  function applyView(){ $('canvas').style.transform=`translate(${pan.x}px,${pan.y}px) scale(${scale})`;drawEdges(); }
  window.addEventListener('message',event=>{const data=event?.data;if(!data||data.type!=='WAN2GP_WORKFLOWS_V2')return;if(data.action==='runtime_previews'){runtimePreviews=Array.isArray(data.previews)?data.previews:[];render();}if(data.action==='node_status'){executionState=data.status&&typeof data.status==='object'?data.status:{nodes:{}};render();}});
  function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));}
  function escapeAttr(value){return escapeHtml(value);}
  document.addEventListener('click',event=>{const action=event.target.closest('[data-action]')?.dataset.action;if(!action)return;if(action==='add')addNode(event.target.closest('[data-action]').dataset.node);if(action==='add-block')insertBlock(event.target.closest('[data-action]').dataset.blockId);if(action==='delete-block')deleteBlock(event.target.closest('[data-action]').dataset.blockId);if(action==='group')createGroup();if(action==='save-block')saveSelectedBlock();if(action==='ungroup')ungroup();if(action==='save'||action==='run')emit(action);if(action==='fit')fit();if(action==='delete')deleteSelection();});
  $('nodeSearch').addEventListener('input',renderPalette);$('viewport').addEventListener('pointerdown',event=>{if(event.button!==0||event.target.closest?.('.node,.port,.edge'))return;panning={x:event.clientX,y:event.clientY,px:pan.x,py:pan.y};$('viewport').setPointerCapture?.(event.pointerId);$('viewport').classList.add('dragging');});
  window.addEventListener('pointermove',event=>{if(dragging){const dx=(event.clientX-dragging.startX)/scale,dy=(event.clientY-dragging.startY)/scale;dragging.positions.forEach(item=>{item.node.position.x=item.x+dx;item.node.position.y=item.y+dy;const element=document.querySelector(`[data-node-id="${CSS.escape(item.node.id)}"]`);if(element){element.style.left=`${item.node.position.x}px`;element.style.top=`${item.node.position.y}px`;}});updateGroupGeometry();drawEdges();}else if(panning){pan.x=panning.px+event.clientX-panning.x;pan.y=panning.py+event.clientY-panning.y;applyView();}});window.addEventListener('pointerup',()=>{dragging=null;panning=null;$('viewport').classList.remove('dragging');});
  $('viewport').addEventListener('wheel',event=>{event.preventDefault();scale=Math.max(.45,Math.min(2.2,scale*(event.deltaY<0?1.08:.92)));applyView();},{passive:false});
  window.addEventListener('keydown',event=>{if((event.key==='Delete'||event.key==='Backspace')&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))deleteSelection();if(event.key.toLowerCase()==='g'&&(event.ctrlKey||event.metaKey)){event.preventDefault();createGroup();}if(event.key==='Escape'){pending=null;selectedEdge=null;render();}});
  renderPalette();render();if(window.W2GP_STATUS)$('status').textContent=window.W2GP_STATUS;
})();
