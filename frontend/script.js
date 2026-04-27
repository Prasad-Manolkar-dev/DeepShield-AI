// =======================
// UPLOAD + ANALYSIS
// =======================

async function uploadFile() {

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Select a file");
        return;
    }

    const preview = document.getElementById("previewContainer");
    const analysis = document.getElementById("analysisSection");

    preview.innerHTML = "";
    analysis.classList.remove("hidden");

    const url = URL.createObjectURL(file);

    if (file.type.startsWith("video")) {
        const video = document.createElement("video");
        video.src = url;
        video.controls = true;
        video.autoplay = true;
        video.className = "preview-box";
        preview.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.src = url;
        img.className = "preview-box";
        preview.appendChild(img);
    }

    const bar = document.getElementById("bar");
    const text = document.getElementById("value1");
    const result = document.getElementById("resultText");

    bar.style.width = "0%";
    text.innerText = "Analyzing...";
    result.innerText = "";

    let progress = 0;

    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress++;
            bar.style.width = progress + "%";
            text.innerText = progress + "%";
        }
    }, 30);

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://127.0.0.1:8000/predict/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        clearInterval(progressInterval);

        let realScore = Math.round(data.average_probability * 100);
        let fakeScore = 100 - realScore;

        bar.style.width = "100%";

        if (realScore >= fakeScore) {
            text.innerText = realScore + "%";
            result.innerText = realScore + "% REAL";
        } else {
            text.innerText = fakeScore + "%";
            result.innerText = fakeScore + "% FAKE";
        }

    } catch (err) {
        clearInterval(progressInterval);
        alert("Backend error");
    }
}


// =======================
// THREE JS SETUP
// =======================

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.z = 4;

const renderer = new THREE.WebGLRenderer({ alpha:true, antialias:true });
renderer.setSize(window.innerWidth, window.innerHeight);

document.getElementById("threeContainer").appendChild(renderer.domElement);


// LIGHT
const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(2,2,5);
scene.add(light);


// =======================
// MODELS (LOW + HIGH)
// =======================

let wireframeModel, lowPolyModel, solidModel;

const loader = new THREE.GLTFLoader();

// HIGH (REAL)
loader.load("model/face_high.glb", (gltf) => {

    solidModel = gltf.scene;

    solidModel.rotation.y = Math.PI;
    solidModel.scale.set(1.2,1.2,1.2);
    solidModel.position.set(0,-0.5,0);

    solidModel.traverse((c)=>{
        if(c.isMesh){
            c.material = c.material.clone();
            c.material.transparent = true;
            c.material.opacity = 0;
        }
    });

    scene.add(solidModel);
});

// LOW
loader.load("model/face_low.glb", (gltf) => {

    lowPolyModel = gltf.scene;

    lowPolyModel.rotation.y = Math.PI;
    lowPolyModel.scale.set(1.2,1.2,1.2);
    lowPolyModel.position.set(0,-0.5,0);

    lowPolyModel.traverse((c)=>{
        if(c.isMesh){
            c.material = new THREE.MeshStandardMaterial({
                color:0x4aa3ff,
                flatShading:true,
                transparent:true,
                opacity:0
            });
        }
    });

    scene.add(lowPolyModel);

    // wireframe
    wireframeModel = lowPolyModel.clone();

    wireframeModel.traverse((c)=>{
        if(c.isMesh){
            c.material = new THREE.MeshBasicMaterial({
                color:0x6EC1FF,
                wireframe:true
            });
        }
    });

    scene.add(wireframeModel);

    startCycle();
});


// =======================
// TRANSITION
// =======================

function startCycle(){

    function run(){

        setOpacity(wireframeModel,1);
        setOpacity(lowPolyModel,0);
        setOpacity(solidModel,0);

        setTimeout(()=>{
            fade(wireframeModel,1,0);
            fade(lowPolyModel,0,1);
        },4000);

        setTimeout(()=>{
            fade(lowPolyModel,1,0);
            fade(solidModel,0,1);
        },8000);
    }

    run();
    setInterval(run,12000);
}


// =======================
// HELPERS
// =======================

function setOpacity(model,val){
    model.traverse(c=>{
        if(c.material){
            c.material.transparent = true;
            c.material.opacity = val;
        }
    });
}

function fade(model,from,to){

    let t=0;

    const interval=setInterval(()=>{

        t+=0.02;
        if(t>=1) clearInterval(interval);

        const v=from+(to-from)*t;

        model.traverse(c=>{
            if(c.material){
                c.material.opacity=v;
            }
        });

        if(Math.random()<0.1) glitch();

    },30);
}


// =======================
// GLITCH
// =======================

function glitch(){

    const el = renderer.domElement;

    el.style.transform = `translate(${(Math.random()-0.5)*10}px, ${(Math.random()-0.5)*10}px)`;

    setTimeout(()=>{ el.style.transform="none"; },80);
}


// =======================
// ANIMATION
// =======================

function animate(){
    requestAnimationFrame(animate);

    if(wireframeModel){
        wireframeModel.rotation.y+=0.0015;
        lowPolyModel.rotation.y+=0.0015;
        solidModel.rotation.y+=0.0015;
    }

    renderer.render(scene,camera);
}

animate();


// =======================
// RESPONSIVE
// =======================

window.addEventListener("resize",()=>{
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
});