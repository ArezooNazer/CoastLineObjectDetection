# CoastLineObjectDetection

In this project we detected common objects on sea shore, using pre-trained Caffe model.

<h3>Structure</h3>
<ul>
  <li>
  <b>coastlineObjectDetection.py :</b> The main program.
  </li>
  <li>
  <b>IntersectionOverUnion.py :</b> The code for calculate the Intersection over Union (IoU) for results of object detection.
  </li>
  <li>
  <b>7grandTruth.py :</b> The same as IntersectionOverUnion.py.
  </li>
  <li>
  <b>video folder :</b> The videos for test.
  </li>
  <li>
  <b>grandtruth folder :</b> The grand truth images .
    <ul>
    <li>iouResults : The results of grand truth</li>
    </ul>
  </li>
   <li>
  <b>MobileNetSSD_deploy.caffemodel :</b> The trained weigths are saved in this file.
  </li>
   <li>
  <b>MobileNetSSD_deploy.prototxt.txt :</b> The network structure is defined in this file.
  </li>
</ul>

<h3> Model </h3>
We used Caffe framework for desing model. The trained weigths are saved in <b>MobileNetSSD_deploy.caffemodel</b> and the network structure is defined in 
<b>MobileNetSSD_deploy.prototxt.txt</b>.

<h3>Dataset</h3>
<ul>
  <li>
    <b>train :</b> Coco dataset.
  </li>
    <li>
    <b>test :</b> Videos captured with iphon7 form Gilan sea shores which is one of the province of Iran.
  </li>
 </ul>
 
 <h3>Results</h3>
 ![IoU result](https://github.com/ArezooNazer/CoastLineObjectDetection/blob/master/grandtruth/iouResult/result1.jpg)
 ![IoU result](https://github.com/ArezooNazer/CoastLineObjectDetection/blob/master/grandtruth/iouResult/result2.jpg)
 ![IoU result](https://github.com/ArezooNazer/CoastLineObjectDetection/blob/master/grandtruth/iouResult/result3.jpg)
 
 <h3> 
  Thanks to https://www.pyimagesearch.com/

  </h3>
  
