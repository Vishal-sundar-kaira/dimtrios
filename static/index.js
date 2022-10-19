let image=document.getElementById("url")
var uploaded_image="";
image.addEventListener("change",function(){
    const reader= new FileReader();
    reader.addEventListener("load",()=>{
        uploaded_image=reader.result;
        document.getElementById("displayimage").style.backgroundImage=`url(${uploaded_image})`;
    })
    reader.readAsDataURL(this.files[0]);
})
let text=document.getElementById("url")
text.addEventListener('change', function() { 
    var fr=new FileReader();
    fr.onload=function(){
        document.getElementById('output').textContent+=fr.result;
    }
        
    fr.readAsText(this.files[0]);
})