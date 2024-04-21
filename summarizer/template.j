<!DOCTYPE html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width">
 <title>Summary of {{title}}</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
</head>
<body>
 <div class="mt-2 container-sm">
  <div class="vstack gap-3">
   <h1>Summary of <a href="{{video_url}}" target="_blank">{{title}}</a></h1>
   {% for i, hr_summary in enumerate(summaries) %}
   <section>
    <h3><a href="{{time_url(video_url, i, 0)}}" target="_blank">{{"{:02d}".format(i)}}:00:00</a></h3>
    <p>{{hr_summary.overall}}</p>
    <ul>
    {% for j, sec in enumerate(hr_summary['parts']) %}
    {% if sec != "" %}
     <li>
      <strong><a href="{{time_url(video_url, i, j)}}" target="_blank">{{"{:02d}".format(i)}}:{{"{:02d}".format(j * 5)}}:00</a></strong>
      {{sec}}
     </li>
    {% endif %}
    {% endfor %}
    </ul>
   </section>
   {% endfor %}
  </div>
 </div>
</body>
</html>

