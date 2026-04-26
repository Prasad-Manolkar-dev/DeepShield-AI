// =======================
// UPLOAD + PREDICTION
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

        startTransition();

    } catch (err) {
        clearInterval(progressInterval);
        alert("Error connecting to backend");
    }
}

// =======================
// THREE JS SETUP
// =======================

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

camera.position.z = 4;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);

document.getElementById("threeContainer").appendChild(renderer.domElement);


// LIGHT
const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(2, 2, 5);
scene.add(light);


// =======================
// MODEL
// =======================

let wireframeModel, lowPolyModel, solidModel;

const loader = new THREE.GLTFLoader();

loader.load("model/face.glb", function (gltf) {

    const base = gltf.scene;

    // 🔥 SIMPLE STABLE ORIENTATION (DO NOT CHANGE)
    base.rotation.y = Math.PI;

    base.scale.set(1.2, 1.2, 1.2);
    base.position.set(0, -0.5, 0);


    // =========================
    // CLONES
    // =========================
    wireframeModel = base.clone();
    lowPolyModel = base.clone();
    solidModel = base.clone();


    // =========================
    // WIREFRAME
    // =========================
    wireframeModel.traverse((child) => {
        if (child.isMesh) {
            child.material = new THREE.MeshBasicMaterial({
                color: 0x6EC1FF,
                wireframe: true
            });
        }
    });


    // =========================
    // LOW POLY
    // =========================
    lowPolyModel.traverse((child) => {
        if (child.isMesh) {
            child.material = new THREE.MeshStandardMaterial({
                color: 0x6EC1FF,
                flatShading: true,
                transparent: true,
                opacity: 0
            });
        }
    });


    // =========================
    // SOLID (FIXED TEXTURE)
    // =========================
    solidModel.traverse((child) => {
        if (child.isMesh) {

            // 🔥 KEEP ORIGINAL TEXTURE
            const originalMaterial = child.material;

            child.material = originalMaterial.clone();

            child.material.transparent = true;
            child.material.opacity = 0;
        }
    });


    scene.add(wireframeModel);
    scene.add(lowPolyModel);
    scene.add(solidModel);

    startAutoCycle();
});


// =======================
// TRANSITION SYSTEM
// =======================

function startAutoCycle() {

    function runCycle() {

        setOpacity(wireframeModel, 1);
        setOpacity(lowPolyModel, 0);
        setOpacity(solidModel, 0);

        // 0–4s → wireframe
        setTimeout(() => {
            fade(wireframeModel, 1, 0);
            fade(lowPolyModel, 0, 1);
        }, 4000);

        // 4–8s → low poly
        setTimeout(() => {
            fade(lowPolyModel, 1, 0);
            fade(solidModel, 0, 1);
        }, 8000);

    }

    runCycle();
    setInterval(runCycle, 12000);
}


// =======================
// HELPERS
// =======================

function setOpacity(model, value) {
    model.traverse((child) => {
        if (child.material) {
            child.material.transparent = true;
            child.material.opacity = value;
        }
    });
}

function fade(model, from, to) {

    let t = 0;

    const interval = setInterval(() => {

        t += 0.02;

        if (t >= 1) clearInterval(interval);

        const value = from + (to - from) * t;

        model.traverse((child) => {
            if (child.material) {
                child.material.opacity = value;
            }
        });

    }, 30);
}


// =======================
// ANIMATION
// =======================

function animate() {
    requestAnimationFrame(animate);

    if (wireframeModel) {
        wireframeModel.rotation.y += 0.0015;
        lowPolyModel.rotation.y += 0.0015;
        solidModel.rotation.y += 0.0015;
    }

    renderer.render(scene, camera);
}

animate();


// =======================
// RESPONSIVE
// =======================

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});